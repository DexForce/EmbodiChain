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
import os
import random
from typing import TYPE_CHECKING, Literal, Union, List, Dict, Sequence

from embodichain.lab.sim.objects import RigidObject, Articulation, Robot
from embodichain.lab.sim.sensors import Camera, StereoCamera
from embodichain.lab.sim.types import EnvObs
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers.events import resolve_dict
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.utils import logger
from embodichain.utils.math import quat_from_matrix

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


def get_rigid_object_pose(
    env: EmbodiedEnv,
    obs: EnvObs,
    entity_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Get the world poses of the rigid objects in the environment.

    If the rigid object with the specified UID does not exist in the environment,
    a zero tensor will be returned.

    Args:
        env: The environment instance.
        obs: The observation dictionary.
        entity_cfg: The configuration of the scene entity.

    Returns:
        A tensor of shape (num_envs, 4, 4) representing the world poses of the rigid objects.
    """

    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return torch.zeros((env.num_envs, 4, 4), dtype=torch.float32)

    obj = env.sim.get_rigid_object(entity_cfg.uid)

    return obj.get_local_pose(to_matrix=True)


def normalize_robot_joint_data(
    env: EmbodiedEnv,
    data: torch.Tensor,
    joint_ids: Sequence[int],
    limit: Literal["qpos_limits", "qvel_limits"] = "qpos_limits",
) -> torch.Tensor:
    """Normalize the robot joint positions to the range of [0, 1] based on the joint limits.

    Args:
        env: The environment instance.
        obs: The observation dictionary.
        joint_ids: The indices of the joints to be normalized.
        limit: The type of joint limits to be used for normalization. Options are:
            - `qpos_limits`: Use the joint position limits for normalization.
            - `qvel_limits`: Use the joint velocity limits for normalization.
    """

    robot = env.robot

    # shape of target_limits: (num_envs, len(joint_ids), 2)
    target_limits = getattr(robot.body_data, limit)[:, joint_ids, :]

    # normalize the joint data to the range of [0, 1]
    data[:, joint_ids] = (data[:, joint_ids] - target_limits[:, :, 0]) / (
        target_limits[:, :, 1] - target_limits[:, :, 0]
    )

    return data


def get_sensor_pose_in_robot_frame(
    env: EmbodiedEnv,
    obs: EnvObs,
    entity_cfg: SceneEntityCfg,
    robot_uid: str | None = None,
) -> torch.Tensor:
    """Get the pose of a sensor in the robot's base coordinate frame.

    Args:
        env: The environment instance.
        obs: The observation dictionary.
        entity_cfg: The configuration of the sensor entity.
        robot_uid: The uid of the robot. If None, uses the default robot from env.

    Returns:
        A tensor of shape (num_envs, 7) representing the sensor pose in robot coordinates as [x, y, z, qw, qx, qy, qz].
    """
    # Get robot base pose in world frame
    robot = env.sim.get_robot(robot_uid) if robot_uid else env.robot
    robot_pose = robot.get_local_pose(to_matrix=True)
    robot_pose_inv = torch.linalg.inv(robot_pose)

    # Get sensor pose in world frame
    sensor: Union[Camera, StereoCamera] = env.sim.get_sensor(entity_cfg.uid)
    if sensor is None:
        logger.log_error(
            f"Sensor with UID '{entity_cfg.uid}' not found in the simulation."
        )

    cam_pose = sensor.get_arena_pose(to_matrix=True)

    # Compute sensor pose in robot coordinate frame: T_robot_cam = inv(T_world_robot) @ T_world_cam
    cam_in_robot = torch.matmul(robot_pose_inv, cam_pose)

    # Convert (num_envs, 4, 4) to (num_envs, 7): [x, y, z, qw, qx, qy, qz]
    xyz = cam_in_robot[:, :3, 3]
    quat = quat_from_matrix(cam_in_robot[:, :3, :3])
    pose = torch.cat([xyz, quat], dim=-1)

    return pose


def get_sensor_intrinsics(
    env: EmbodiedEnv,
    obs: EnvObs,
    entity_cfg: SceneEntityCfg,
    is_right: bool = False,
) -> torch.Tensor:
    """Get the intrinsic matrix of a sensor (camera).

    Args:
        env: The environment instance.
        obs: The observation dictionary.
        entity_cfg: The configuration of the sensor entity.
        is_right: Whether to return the right camera intrinsics for stereo cameras.
            Defaults to False (left camera). Ignored for monocular cameras.

    Returns:
        A tensor of shape (num_envs, 3, 3) representing the camera intrinsics.
    """
    sensor = env.sim.get_sensor(entity_cfg.uid)
    if sensor is None:
        logger.log_error(
            f"Sensor with UID '{entity_cfg.uid}' not found in the simulation."
        )
    if isinstance(sensor, StereoCamera):
        left_intrinsics, right_intrinsics = sensor.get_intrinsics()
        return right_intrinsics if is_right else left_intrinsics
    elif isinstance(sensor, Camera):
        return sensor.get_intrinsics()  # (num_envs, 3, 3)
    else:
        logger.log_error(f"Sensor '{entity_cfg.uid}' is not Camera or StereoCamera.")
        return torch.zeros((env.num_envs, 3, 3), dtype=torch.float32)


def compute_semantic_mask(
    env: EmbodiedEnv,
    obs: EnvObs,
    entity_cfg: SceneEntityCfg,
    foreground_uids: Sequence[str],
    is_right: bool = False,
) -> torch.Tensor:
    """Compute the semantic mask for the specified scene entity.

    Note:
        The semantic mask is defined as (B, H, W, 3) where the three channels represents:
        - robot channel: the instance id of the robot is set to 1 (0 if not robot)
        - background channel: the instance id of the background is set to 1 (0 if not background)
        - foreground channel: the instance id of the foreground objects is set to 1 (0 if not foreground)

    Args:
        env: The environment instance.
        obs: The observation dictionary.
        entity_cfg: The configuration of the scene entity.
        foreground_uids: The list of uids for the foreground objects.
        is_right: Whether to use the right camera for stereo cameras. Default is False.
            Only applicable if the sensor is a StereoCamera.

    Returns:
        A tensor of shape (num_envs, height, width) representing the semantic mask.
    """
    from embodichain.data.enum import SemanticMask

    sensor: Union[Camera, StereoCamera] = env.sim.get_sensor(entity_cfg.uid)
    if sensor.cfg.enable_mask is False:
        logger.log_error(
            f"Sensor '{entity_cfg.uid}' does not have mask enabled. Please enable the mask in the sensor configuration."
        )

    if isinstance(sensor, StereoCamera) and is_right:
        mask = obs["sensor"][entity_cfg.uid]["mask_right"]
    else:
        mask = obs["sensor"][entity_cfg.uid]["mask"]

    robot_uids = env.robot.get_user_ids()

    mask_exp = mask.unsqueeze(-1)

    robot_uids_exp = robot_uids.unsqueeze_(1).unsqueeze_(1)

    robot_mask = (mask_exp == robot_uids_exp).any(-1).squeeze_(-1)

    asset_uids = env.sim.asset_uids
    foreground_assets = [
        env.sim.get_asset(uid) for uid in foreground_uids if uid in asset_uids
    ]

    # cat assets uid (num_envs, n) into dim 1
    foreground_uids = torch.cat(
        [
            (
                asset.get_user_ids().unsqueeze(1)
                if asset.get_user_ids().dim() == 1
                else asset.get_user_ids()
            )
            for asset in foreground_assets
        ],
        dim=1,
    )

    foreground_uids_exp = foreground_uids.unsqueeze_(1).unsqueeze_(1)

    foreground_mask = (mask_exp == foreground_uids_exp).any(-1).squeeze_(-1)

    background_mask = ~(robot_mask | foreground_mask).squeeze_(-1)

    masks = [None, None, None]
    masks_ids = [member.value for member in SemanticMask]
    assert len(masks) == len(
        masks_ids
    ), "Different length of mask slots and SemanticMask Enum {}.".format(masks_ids)
    mask_id_to_label = {
        SemanticMask.BACKGROUND.value: background_mask,
        SemanticMask.FOREGROUND.value: foreground_mask,
        SemanticMask.ROBOT.value: robot_mask,
    }
    for mask_id in masks_ids:
        masks[mask_id] = mask_id_to_label[mask_id]
    return torch.stack(masks, dim=-1)


def get_robot_eef_pose(
    env: "EmbodiedEnv",
    obs: EnvObs,
    part_name: str | None = None,
    position_only: bool = False,
) -> torch.Tensor:
    """Get robot end-effector pose using forward kinematics.

    Args:
        env: The environment instance.
        obs: The observation dictionary.
        part_name: The name of the control part. If None, uses default part.
        position_only: If True, returns only position (3D). If False, returns full pose (4x4 matrix).

    Returns:
        A tensor of shape (num_envs, 3) if position_only=True, or (num_envs, 4, 4) otherwise.
    """
    robot = env.robot

    if part_name is not None:
        joint_ids = robot.get_joint_ids(part_name)
        qpos = robot.body_data.qpos[:, joint_ids]
        ee_pose = robot.compute_fk(name=part_name, qpos=qpos, to_matrix=True)
    else:
        qpos = robot.get_qpos()
        ee_pose = robot.compute_fk(qpos=qpos, to_matrix=True)

    if position_only:
        return ee_pose[:, :3, 3]
    return ee_pose


def target_position(
    env: "EmbodiedEnv",
    obs: EnvObs,
    target_pose_key: str = "goal_pose",
) -> torch.Tensor:
    """Get virtual target position from env state."""
    state_attr = f"_{target_pose_key}s"
    if hasattr(env, state_attr):
        target_poses = getattr(env, state_attr)
        return target_poses[:, :3, 3]
    return torch.zeros(env.num_envs, 3, device=env.device)


class compute_exteroception(Functor):
    """Compute the exteroception for the observation space.

    The exteroception is currently defined as a set of keypoints around a reference pose, which are prjected from 3D
    space to 2D image plane.
    The reference pose can derive from the following sources:
        - Pose from robot control part (e.g., end-effector, usually tcp pose)
        - Object affordance pose (e.g., handle pose of a mug or a pick pose of a cube)

    Therefore, the exteroception are defined in the camera-like sensor, for example.
    descriptor = {
        "cam_high": [
            {
                "type": "affordance",
                "obj_uid": "obj1",
                "key": "grasp_pose",
                "is_arena_coord": True
            },
            {
                "type": "affordance",
                "obj_uid": "obj1",
                "key": "place_pose",
            },
            {
                "type": "robot",
                "control_part": "left_arm",
            },
            {
                "type": "robot",
                "control_part": "right_arm",
            }
        ],
        ...
    }

    Explanation of the parameters:
        - The key of the dictionary is the sensor uid.
        - The value is another dictionary, where the key is the source type, and the value is a dictionary of parameters.
        - For `affordance` source type, the parameters are:
            - `obj_uid`: The uid of the object to get the affordance pose from.
            - `key`: The key of the affordance pose in the affordance data.
            - `is_arena_coord`: Whether the affordance pose is in the arena coordinate system. Default is False.
        - For `robot` source type, the parameters are:
            - `control_part`: The control part of the robot to get the pose from.
    """

    def __init__(
        self,
        cfg: FunctorCfg,
        env: EmbodiedEnv,
    ):
        super().__init__(cfg, env)

        if self._env.num_envs != 1:
            logger.log_error(
                f"Exteroception functor only supported env with 'num_envs=1' but got 'num_envs={self._env.num_envs}'. Please check again."
            )

        self._valid_source = ["robot", "affordance"]

    @staticmethod
    def shift_pose(pose: torch.Tensor, axis: int, shift: float) -> torch.Tensor:
        """Shift the pose along the specified axis by the given amount.

        Args:
            pose: The original pose tensor of shape (B, 4, 4).
            axis: The axis along which to shift (0 for x, 1 for y, 2 for z).
            shift: The amount to shift along the specified axis.
        """
        shift_pose = torch.linalg.inv(pose)
        shift_pose[:, axis, -1] += shift
        shift_pose = torch.linalg.inv(shift_pose)
        return shift_pose

    @staticmethod
    def expand_pose(
        pose: torch.Tensor,
        x_interval: float,
        y_interval: float,
        kpnts_number: int,
        ref_pose: torch.Tensor = None,
    ) -> torch.Tensor:
        """Expand pose with keypoints along x and y axes.

        Args:
            pose: The original pose tensor of shape (B, 4, 4).
            x_interval: The interval for expanding along x-axis.
            y_interval: The interval for expanding along y-axis.
            kpnts_number: Number of keypoints to generate for each axis.
            ref_pose: Reference pose tensor of shape (B, 4, 4). If None, uses identity matrix.

        Returns:
            Expanded poses tensor of shape (B, 1 + 2*kpnts_number, 4, 4).
        """
        batch_size = pose.shape[0]
        device = pose.device

        # Create default reference pose if not provided
        if ref_pose is None:
            ref_pose = (
                torch.eye(4, device=device).unsqueeze_(0).repeat(batch_size, 1, 1)
            )

        # Start with the original pose transformed by ref_pose
        ret = [ref_pose @ pose]

        # Generate x-axis offsets and expand poses
        # TODO: only support 1 env
        xoffset = torch.linspace(-x_interval, x_interval, kpnts_number, device=device)
        for x_shift in xoffset:
            shifted_pose = compute_exteroception.shift_pose(pose, 0, x_shift.item())
            x_expanded = ref_pose @ shifted_pose
            ret.append(x_expanded)

        # Generate y-axis offsets and expand poses
        # TODO: only support 1 env
        yoffset = torch.linspace(-y_interval, y_interval, kpnts_number, device=device)
        for y_shift in yoffset:
            shifted_pose = compute_exteroception.shift_pose(pose, 1, y_shift.item())
            y_expanded = ref_pose @ shifted_pose
            ret.append(y_expanded)

        # Stack all poses along a new dimension
        return torch.stack(ret, dim=1)

    @staticmethod
    def _project_3d_to_2d(
        cam_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
        target_poses: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Project 3D poses to 2D image plane.

        Args:
            cam_pose: Camera pose of in arena frame of shape (B, 4, 4).
            intrinsics: Camera intrinsic matrix of shape (B, 3, 3).
            height: Image height.
            width: Image width.
            target_poses: 3D poses of shape (B, N, 4, 4).
            normalize: Whether to normalize the projected points to [0, 1] range.

        Returns:
            Projected 2D points of shape (B, N, 2).
        """
        batch_size, num_poses = target_poses.shape[:2]

        # Convert to opencv coordinate system
        cam_pose[:, :3, 1] = -cam_pose[:, :3, 1]
        cam_pose[:, :3, 2] = -cam_pose[:, :3, 2]

        # Expand cam_pose_inv and intrinsics to match target_poses batch dimension
        cam_pose_inv = torch.linalg.inv(cam_pose)  # (B, 4, 4)
        cam_pose_inv_expanded = cam_pose_inv.unsqueeze(1).expand(
            -1, num_poses, -1, -1
        )  # (B, N, 4, 4)
        cam_pose_inv_reshaped = cam_pose_inv_expanded.reshape(-1, 4, 4)  # (B*N, 4, 4)

        intrinsics_expanded = intrinsics.unsqueeze(1).expand(
            -1, num_poses, -1, -1
        )  # (B, N, 3, 3)
        intrinsics_reshaped = intrinsics_expanded.reshape(-1, 3, 3)  # (B*N, 3, 3)

        # Reshape target_poses to (B*N, 4, 4)
        target_poses_reshaped = target_poses.reshape(-1, 4, 4)  # (B*N, 4, 4)

        # Transform 3D points to camera coordinates in parallel
        # Extract translation part (position) from target poses: (B*N, 4, 1)
        target_positions = target_poses_reshaped[:, :, 3:4]  # (B*N, 4, 1)

        # Transform to camera coordinates: (B*N, 4, 1)
        cam_positions = cam_pose_inv_reshaped.bmm(target_positions)  # (B*N, 4, 1)
        cam_positions_3d = cam_positions[:, :3, 0]  # (B*N, 3)

        # Project to 2D using intrinsics in parallel
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        z_safe = torch.clamp(cam_positions_3d[:, 2], min=eps)  # (B*N,)

        # Normalize by depth
        normalized_points = cam_positions_3d[:, :2] / z_safe.unsqueeze(-1)  # (B*N, 2)

        # Convert to homogeneous coordinates and apply intrinsics
        normalized_homogeneous = torch.cat(
            [normalized_points, torch.ones_like(normalized_points[:, :1])], dim=-1
        )  # (B*N, 3)
        pixel_coords = intrinsics_reshaped.bmm(
            normalized_homogeneous.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (B*N, 3)

        # Extract 2D coordinates
        points_2d_flat = pixel_coords[:, :2]  # (B*N, 2)

        # Reshape back to (B, N, 2)
        points_2d = points_2d_flat.reshape(batch_size, num_poses, 2)

        # clip to range [0, width] and [0, height]
        points_2d[..., 0] = torch.clamp(points_2d[..., 0], 0, width - 1)
        points_2d[..., 1] = torch.clamp(points_2d[..., 1], 0, height - 1)

        if normalize:
            # Normalize to [0, 1] range
            points_2d[..., 0] /= width
            points_2d[..., 1] /= height

        return points_2d

    def _get_gripper_ratio(
        self, control_part: str, gripper_qpos: torch.Tensor | None = None
    ):
        robot: Robot = self._env.robot
        gripper_max_limit = robot.body_data.qpos_limits[
            :, robot.get_joint_ids(control_part)
        ][:, 0, 1]

        if gripper_qpos is None:
            gripper_qpos = robot.get_qpos()[:, robot.get_joint_ids(control_part)][:, 0]

        return gripper_qpos / gripper_max_limit

    def _get_robot_exteroception(
        self,
        control_part: str | None = None,
        x_interval: float = 0.02,
        y_interval: float = 0.02,
        kpnts_number: int = 12,
        offset: list | torch.Tensor | None = None,
        follow_eef: bool = False,
    ) -> torch.Tensor:
        """Get the robot exteroception poses.

        Args:
            control_part: The part of the robot to use as reference. If None, uses the base.
            x_interval: The interval for expanding along x-axis.
            y_interval: The interval for expanding along y-axis.
            kpnts_number: Number of keypoints to generate for each axis.
            offset: Intrinsic offset that need to be substracted.
            follow_eef: Whether to follow the gripper or not.

        Returns:
            A tensor of shape (num_envs, 1 + 2*kpnts_number, 4, 4) representing the exteroception poses.
        """
        robot: Robot = self._env.robot
        if control_part is not None:
            current_qpos = robot.get_qpos()[:, robot.get_joint_ids(control_part)]
            robot_pose = robot.compute_fk(
                current_qpos, name=control_part, to_matrix=True
            )
            if follow_eef:
                gripper_ratio = self._get_gripper_ratio(
                    control_part.replace("_arm", "_eef")
                )  # TODO: "_eef" hardcode
                # TODO: only support 1 env
                y_interval = (y_interval * gripper_ratio)[0].item()
        else:
            logger.log_error("Not supported Robot without control part yet.")

        if offset is not None:
            offset = torch.as_tensor(
                offset, dtype=torch.float32, device=self._env.device
            )

            if (offset.ndim > 2) or (offset.shape[-1] != 3):
                logger.log_error(
                    f"Only (N, 3) shaped xyz-intrinsic offset supported, got shape {offset.shape}"
                )
            elif offset.ndim == 1:
                offset = offset[None]
            # TODO: This operation may be slow when large scale Parallelization, but when small (num_envs=1) this operation is faster
            robot_pose[:, :3, 3] = robot_pose[:, :3, 3] - torch.einsum(
                "bij,bj->bi", robot_pose[:, :3, :3], offset
            )

        return compute_exteroception.expand_pose(
            robot_pose,
            x_interval,
            y_interval,
            kpnts_number,
        )

    def _get_object_exteroception(
        self,
        uid: str,
        affordance_key: str,
        x_interval: float = 0.02,
        y_interval: float = 0.02,
        kpnts_number: int = 12,
        is_arena_coord: bool = False,
        follow_eef: str | None = None,
    ) -> torch.Tensor:
        """Get the rigid object exteroception poses.

        Args:
            uid: The UID of the object.
            affordance_key: The key of the affordance to use for the object pose.
            x_interval: The interval for expanding along x-axis.
            y_interval: The interval for expanding along y-axis.
            kpnts_number: Number of keypoints to generate for each axis.
            is_arena_coord: Whether to use the arena coordinate system. Default is False.

        Returns:
            A tensor of shape (num_envs, 1 + 2*kpnts_number, 4, 4) representing the exteroception poses.
        """

        obj: RigidObject = self._env.sim.get_rigid_object(uid)
        if obj is None:
            logger.log_error(
                f"Rigid object with UID '{uid}' not found in the simulation."
            )

        if hasattr(self._env, "affordance_datas") is False:
            logger.log_error(
                "Affordance data is not available in the environment. We cannot compute object exteroception."
            )

        if affordance_key not in self._env.affordance_datas:
            # TODO: should this default behavior be warned?
            # logger.log_warning(
            #     f"Affordance key '{affordance_key}' not found in the affordance data, using identity pose.."
            # )
            pass

        affordance_pose = torch.as_tensor(
            self._env.affordance_datas.get(
                affordance_key, torch.eye(4).repeat(self._env.num_envs, 1, 1)
            ),
            dtype=torch.float32,
        )
        if affordance_pose.ndim < 3:
            affordance_pose = affordance_pose.repeat(self._env.num_envs, 1, 1)

        ref_pose = None if is_arena_coord else obj.get_local_pose(to_matrix=True)

        if follow_eef is not None:
            gripper_ratio = self._get_gripper_ratio(control_part=follow_eef)
            # TODO: only support 1 env
            y_interval = (y_interval * gripper_ratio)[0].item()

        return compute_exteroception.expand_pose(
            affordance_pose,
            x_interval,
            y_interval,
            kpnts_number,
            ref_pose=ref_pose,
        )

    def _check_source_valid(self, source: str) -> bool:
        if source not in self._valid_source:
            logger.log_error(
                f"Invalid exteroception source '{source}'. Supported sources are {self._valid_source}."
            )
        return True

    def __call__(
        self,
        env: EmbodiedEnv,
        obs: EnvObs,
        descriptor: Dict[str, Dict[str, str]],
        x_interval: float = 0.02,
        y_interval: float = 0.02,
        kpnts_number: int = 12,
        groups: int = 6,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute the exteroception poses based on the asset type.

        Args:
            descriptor: The observation dictionary.

        Returns:
            A dictionary containing the exteroception poses with key 'exteroception'.
        """

        exteroception = {}
        descriptor = resolve_dict(self._env, descriptor)
        for sensor_uid, sources in descriptor.items():
            sensor: Union[Camera, StereoCamera] = self._env.sim.get_sensor(sensor_uid)
            if sensor is None:
                logger.log_error(
                    f"Sensor with UID '{sensor_uid}' not found in the simulation."
                )

            if not isinstance(sensor, (Camera, StereoCamera)):
                logger.log_error(
                    f"Sensor with UID '{sensor_uid}' is not a Camera or StereoCamera."
                )

            height, width = sensor.cfg.height, sensor.cfg.width

            exteroception[sensor_uid] = {}
            taget_pose_list = []
            for source in sources:
                source_type = source["type"]
                self._check_source_valid(source_type)

                if source_type == "robot":
                    target_pose = self._get_robot_exteroception(
                        control_part=source["control_part"],
                        x_interval=x_interval,
                        y_interval=y_interval,
                        kpnts_number=kpnts_number,
                        offset=source.get("offset", None),
                        follow_eef=source.get("follow_eef", False),
                    )
                elif source_type == "affordance":
                    target_pose = self._get_object_exteroception(
                        uid=source["obj_uid"],
                        affordance_key=source["key"],
                        x_interval=x_interval,
                        y_interval=y_interval,
                        kpnts_number=kpnts_number,
                        is_arena_coord=source["is_arena_coord"],
                        follow_eef=source.get("follow_eef", None),
                    )
                else:
                    logger.log_error(
                        f"Unsupported exteroception source '{source_type}'. Supported sources are 'robot' and 'affordance."
                    )
                taget_pose_list.append(target_pose)

            target_poses = torch.cat(taget_pose_list, dim=1)
            if target_poses.shape[1] / (2 * kpnts_number + 1) != groups:
                logger.log_error(
                    f"Exteroception groups number mismatch. Expected {groups}, but got {int(target_poses.shape[1] / (2 * kpnts_number + 1))}."
                )

            if isinstance(sensor, StereoCamera):
                intrinsics, right_intrinsics = sensor.get_intrinsics()
                left_arena_pose, right_arena_pose = sensor.get_left_right_arena_pose()
                projected_kpnts = compute_exteroception._project_3d_to_2d(
                    left_arena_pose,
                    intrinsics,
                    height,
                    width,
                    target_poses,
                )
                exteroception[sensor_uid]["l"] = projected_kpnts

                projected_kpnts = compute_exteroception._project_3d_to_2d(
                    right_arena_pose,
                    right_intrinsics,
                    height,
                    width,
                    target_poses,
                )
                exteroception[sensor_uid]["r"] = projected_kpnts
            else:
                intrinsics = sensor.get_intrinsics()
                projected_kpnts = compute_exteroception._project_3d_to_2d(
                    sensor.get_arena_pose(to_matrix=True),
                    intrinsics,
                    height,
                    width,
                    target_poses,
                )
                exteroception[sensor_uid] = projected_kpnts

        return exteroception
