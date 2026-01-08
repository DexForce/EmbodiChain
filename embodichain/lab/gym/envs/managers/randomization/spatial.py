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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Union, List

from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.utils.math import sample_uniform, matrix_from_euler
from embodichain.utils import logger


if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


def get_random_pose(
    init_pos: torch.Tensor,
    init_rot: torch.Tensor,
    position_range: tuple[list[float], list[float]] | None = None,
    rotation_range: tuple[list[float], list[float]] | None = None,
    relative_position: bool = True,
    relative_rotation: bool = False,
) -> torch.Tensor:
    """Generate a random pose based on the initial position and rotation.

    Args:
        init_pos (torch.Tensor): The initial position tensor of shape (num_instance, 3).
        init_rot (torch.Tensor): The initial rotation tensor of shape (num_instance, 3, 3).
        position_range (tuple[list[float], list[float]] | None): The range for the position randomization.
        rotation_range (tuple[list[float], list[float]] | None): The range for the rotation randomization.
            The rotation is represented as Euler angles (roll, pitch, yaw) in degree.
        relative_position (bool): Whether to randomize the position relative to the initial position. Default is True.
        relative_rotation (bool): Whether to randomize the rotation relative to the initial rotation. Default is False.

    Returns:
        torch.Tensor: The generated random pose tensor of shape (num_instance, 4, 4).
    """

    num_instance = init_pos.shape[0]
    pose = (
        torch.eye(4, dtype=torch.float32, device=init_pos.device)
        .unsqueeze_(0)
        .repeat(num_instance, 1, 1)
    )
    pose[:, :3, :3] = init_rot
    pose[:, :3, 3] = init_pos

    if position_range:

        pos_low = torch.tensor(position_range[0], device=init_pos.device)
        pos_high = torch.tensor(position_range[1], device=init_pos.device)

        random_value = sample_uniform(
            lower=pos_low,
            upper=pos_high,
            size=(num_instance, 3),
            device=init_pos.device,
        )
        if relative_position:
            random_value += init_pos

        pose[:, :3, 3] = random_value

    if rotation_range:

        rot_low = torch.tensor(rotation_range[0], device=init_pos.device)
        rot_high = torch.tensor(rotation_range[1], device=init_pos.device)

        random_value = (
            sample_uniform(
                lower=rot_low,
                upper=rot_high,
                size=(num_instance, 3),
                device=init_pos.device,
            )
            * torch.pi
            / 180.0
        )
        rot = matrix_from_euler(random_value)

        if relative_rotation:
            rot = torch.bmm(init_rot, rot)
        pose[:, :3, :3] = rot

    return pose


def randomize_rigid_object_pose(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    position_range: tuple[list[float], list[float]] | None = None,
    rotation_range: tuple[list[float], list[float]] | None = None,
    relative_position: bool = True,
    relative_rotation: bool = False,
) -> None:
    """Randomize the pose of a rigid object in the environment.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (Union[torch.Tensor, None]): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        position_range (tuple[list[float], list[float]] | None): The range for the position randomization.
        rotation_range (tuple[list[float], list[float]] | None): The range for the rotation randomization.
            The rotation is represented as Euler angles (roll, pitch, yaw) in degree.
        relative_position (bool): Whether to randomize the position relative to the object's initial position. Default is True.
        relative_rotation (bool): Whether to randomize the rotation relative to the object's initial rotation. Default is False.
    """

    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return

    rigid_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)
    num_instance = len(env_ids)

    init_pos = (
        torch.tensor(rigid_object.cfg.init_pos, dtype=torch.float32, device=env.device)
        .unsqueeze_(0)
        .repeat(num_instance, 1)
    )
    init_rot = (
        torch.tensor(rigid_object.cfg.init_rot, dtype=torch.float32, device=env.device)
        * torch.pi
        / 180.0
    )
    init_rot = init_rot.unsqueeze_(0).repeat(num_instance, 1)
    init_rot = matrix_from_euler(init_rot)

    pose = get_random_pose(
        init_pos=init_pos,
        init_rot=init_rot,
        position_range=position_range,
        rotation_range=rotation_range,
        relative_position=relative_position,
        relative_rotation=relative_rotation,
    )

    rigid_object.set_local_pose(pose, env_ids=env_ids)
    rigid_object.clear_dynamics()


def randomize_robot_eef_pose(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    position_range: tuple[list[float], list[float]] | None = None,
    rotation_range: tuple[list[float], list[float]] | None = None,
) -> None:
    """Randomize the initial end-effector pose of a robot in the environment.

    Note:
        - The position and rotation are performed randomization in a relative manner.
        - The current state of eef pose is computed based on the current joint positions of the robot.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (Union[torch.Tensor, None]): The environment IDs to apply the randomization.
        robot_name (str): The name of the robot.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        position_range (tuple[list[float], list[float]] | None): The range for the position randomization.
        rotation_range (tuple[list[float], list[float]] | None): The range for the rotation randomization.
            The rotation is represented as Euler angles (roll, pitch, yaw) in degree.
    """

    def set_random_eef_pose(joint_ids: List[int], robot: Robot) -> None:
        current_qpos = robot.get_qpos()[env_ids][:, joint_ids]
        if current_qpos.dim() == 1:
            current_qpos = current_qpos.unsqueeze_(0)

        current_eef_pose = robot.compute_fk(
            name=part, qpos=current_qpos, to_matrix=True
        )

        new_eef_pose = get_random_pose(
            init_pos=current_eef_pose[:, :3, 3],
            init_rot=current_eef_pose[:, :3, :3],
            position_range=position_range,
            rotation_range=rotation_range,
            relative_position=True,
            relative_rotation=True,
        )

        ret, new_qpos = robot.compute_ik(
            pose=new_eef_pose, name=part, joint_seed=current_qpos
        )

        new_qpos[ret == False] = current_qpos[ret == False]
        robot.set_qpos(new_qpos, env_ids=env_ids, joint_ids=joint_ids)

    robot = env.sim.get_robot(entity_cfg.uid)

    control_parts = entity_cfg.control_parts
    if control_parts is None:
        joint_ids = robot.get_joint_ids()
        set_random_eef_pose(joint_ids, robot)
    else:
        for part in control_parts:
            joint_ids = robot.get_joint_ids(part)
            set_random_eef_pose(joint_ids, robot)

    # simulate 10 steps to let the robot reach the target pose.
    env.sim.update(step=10)


def randomize_robot_qpos(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    qpos_range: tuple[list[float], list[float]] | None = None,
    relative_qpos: bool = True,
    joint_ids: List[int] | None = None,
) -> None:
    """Randomize the initial joint positions of a robot in the environment.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (Union[torch.Tensor, None]): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        qpos_range (tuple[list[float], list[float]] | None): The range for the joint position randomization.
        relative_qpos (bool): Whether to randomize the joint positions relative to the current joint positions. Default is True.
        joint_ids (List[int] | None): The list of joint IDs to randomize. If None, all joints will be randomized.
    """
    if qpos_range is None:
        return

    num_instance = len(env_ids)

    robot = env.sim.get_robot(entity_cfg.uid)

    if joint_ids is None:
        if len(qpos_range[0]) != robot.dof:
            logger.log_error(
                f"The length of qpos_range {len(qpos_range[0])} does not match the robot dof {robot.dof}."
            )
        joint_ids = robot.get_joint_ids()

    qpos = sample_uniform(
        lower=torch.tensor(qpos_range[0], device=env.device),
        upper=torch.tensor(qpos_range[1], device=env.device),
        size=(num_instance, len(joint_ids)),
        device=env.device,
    )

    if relative_qpos:
        current_qpos = robot.get_qpos()[env_ids][:, joint_ids]
        current_qpos += qpos
    else:
        current_qpos = qpos

    robot.set_qpos(qpos=current_qpos, env_ids=env_ids, joint_ids=joint_ids)
    env.sim.update(step=100)


# workspace_sampler class caches reachable poses on init, and samples on call
class workspace_sampler(Functor):
    """
    workspace_sampler samples and caches all reachable object poses in a circular plane for a given robot and object.
    Usage:
        sampler = workspace_sampler(cfg, env)
        pose = sampler(num_instance)
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        """
        Args:
            cfg: dict or object with keys: entity_cfg, robot_name, sampling_radius, control_part, resolution
            env: EmbodiedEnv instance
        """
        from embodichain.lab.sim.utility.workspace_sampler import (
            sample_circular_plane_reachability,
        )

        self.env = env
        self.entity_cfg = cfg.params["entity_cfg"]
        self.robot_name = cfg.params.get("robot_name", "robot")
        self.sampling_radius = cfg.params.get("sampling_radius", 0.2)
        self.control_part = cfg.params.get("control_part", "left_arm")
        self.resolution = cfg.params.get("resolution", 0.01)

        self.rigid_object: RigidObject = env.sim.get_rigid_object(self.entity_cfg.uid)
        robot: Robot = env.sim.get_robot_v2(self.robot_name)

        init_rot = (
            torch.tensor(
                self.rigid_object.cfg.init_rot, dtype=torch.float32, device=env.device
            )
            * torch.pi
            / 180.0
        )
        init_rot_mat = matrix_from_euler(init_rot.unsqueeze(0)).squeeze(0)
        init_pos = torch.tensor(
            self.rigid_object.cfg.init_pos, dtype=torch.float32, device=env.device
        )

        # Get grasp pose in world coordinates
        grasp_poses = getattr(self.rigid_object, "grasp_pose_object", None)
        if isinstance(grasp_poses, torch.Tensor):
            if grasp_poses.dim() == 2:
                grasp_pose_local = grasp_poses  # (4, 4)
            else:
                grasp_pose_local = grasp_poses[0]  # Take first one
        else:
            grasp_pose_local = torch.tensor(
                grasp_poses[0] if isinstance(grasp_poses, list) else grasp_poses,
                dtype=torch.float32,
                device=env.device,
            )

        object_pose = torch.eye(4, dtype=torch.float32, device=env.device)
        object_pose[:3, :3] = init_rot_mat
        object_pose[:3, 3] = init_pos
        grasp_pose_world = object_pose @ grasp_pose_local

        center_xy = (grasp_pose_world[0, 3].item(), grasp_pose_world[1, 3].item())
        z_height = grasp_pose_world[2, 3].item()

        # Precompute reachable poses only once
        _, reachable_poses_np = sample_circular_plane_reachability(
            robot=robot,
            control_part=self.control_part,
            ref_xpos=grasp_pose_world.cpu().numpy(),
            center_xy=center_xy,
            z_height=z_height,
            radius=self.sampling_radius,
            resolution=self.resolution,
        )

        if not reachable_poses_np:
            logger.log_warning("No reachable poses found. Skipping randomization.")
            self.reachable_object_poses = None
            return

        logger.log_info(f"Found {len(reachable_poses_np)} reachable grasp poses.")

        # FIXME: 这个后面应该换成物体的属性
        delta_xpos = env.cfg.action_bank_edit_configs.get(
            "pre_grasp_delta_xpos", [0, 0, 0, 0, 0, 0]
        )

        # For each grasp_pose in reachable_poses_np (already IK-validated),
        # check if the corresponding pre_grasp_pose IK can also be solved
        valid_grasp_poses = []

        for pose in reachable_poses_np:
            grasp_pose_world_sampled = torch.tensor(
                pose, dtype=torch.float32, device=env.device
            )

            # Apply delta to grasp_pose following _apply_delta_to_xpos logic
            pre_grasp_pose = grasp_pose_world_sampled.clone()
            # Apply position delta
            pre_grasp_pose[:3, 3] += torch.tensor(
                delta_xpos[:3], dtype=torch.float32, device=env.device
            )
            # Apply rotation delta
            delta_rpy = (
                torch.tensor(delta_xpos[3:], dtype=torch.float32, device=env.device)
                * torch.pi
                / 180.0
            )
            delta_rot = matrix_from_euler(delta_rpy.unsqueeze(0)).squeeze(0)  # (3,3)
            # Left multiply: delta_rotation * original_rotation
            pre_grasp_pose[:3, :3] = delta_rot @ grasp_pose_world_sampled[:3, :3]

            # Only check pre_grasp_pose IK (grasp_pose is already validated in reachable_poses_np)
            pre_grasp_success, pre_grasp_qpos = robot.compute_ik(
                pose=pre_grasp_pose.unsqueeze(0), name=self.control_part
            )

            # Only keep if pre_grasp IK is successful
            if pre_grasp_success:
                valid_grasp_poses.append(grasp_pose_world_sampled)

        if not valid_grasp_poses:
            logger.log_warning(
                "No valid grasp poses found with solvable pre-grasp IK. Skipping randomization."
            )
            self.reachable_object_poses = None
            return

        logger.log_info(
            f"Filtered to {len(valid_grasp_poses)} valid grasp poses with solvable pre-grasp IK."
        )

        # Precompute all reachable object center poses (4x4 matrices) from valid grasp poses
        grasp_pose_local_inv = torch.inverse(grasp_pose_local)
        object_poses = []
        for grasp_pose_world_sampled in valid_grasp_poses:
            object_pose_sampled = grasp_pose_world_sampled @ grasp_pose_local_inv
            object_poses.append(object_pose_sampled)
        self.reachable_object_poses = torch.stack(
            object_poses
        )  # (M, 4, 4) where M <= N
        self.num_poses = self.reachable_object_poses.shape[0]

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        entity_cfg: dict = None,
        robot_name: str = "robot",
        control_part: str = "left_arm",
        sampling_radius: float = 0.2,
        num_instance: int = 1,
    ):
        """
        Sample object poses from cached reachable object center poses and set them to the object.
        Args:
            num_instance: number of samples to draw
        """
        if self.reachable_object_poses is None:
            logger.log_warning("No reachable poses cached in workspace_sampler.")
            return

        # Randomly sample indices
        idx = torch.randint(0, self.num_poses, (num_instance,))
        pose = self.reachable_object_poses[idx]

        # Set the pose to the rigid object
        self.rigid_object.set_local_pose(pose, env_ids=env_ids)
        self.rigid_object.clear_dynamics()
