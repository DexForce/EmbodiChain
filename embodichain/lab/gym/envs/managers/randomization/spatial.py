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
