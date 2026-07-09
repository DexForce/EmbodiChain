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
from dataclasses import MISSING
from typing import TYPE_CHECKING, Union, List

from embodichain.lab.sim.objects import (
    RigidObject,
    Robot,
    Articulation,
    RigidObjectGroup,
)
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.utils import configclass
from embodichain.utils.math import sample_uniform, matrix_from_euler, matrix_from_quat
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
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    position_range: tuple[list[float], list[float]] | None = None,
    rotation_range: tuple[list[float], list[float]] | None = None,
    relative_position: bool = True,
    relative_rotation: bool = False,
    physics_update_step: int = -1,
) -> None:
    """Randomize the pose of a rigid object in the environment.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | None): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        position_range (tuple[list[float], list[float]] | None): The range for the position randomization.
        rotation_range (tuple[list[float], list[float]] | None): The range for the rotation randomization.
            The rotation is represented as Euler angles (roll, pitch, yaw) in degree.
        relative_position (bool): Whether to randomize the position relative to the object's initial position. Default is True.
        relative_rotation (bool): Whether to randomize the rotation relative to the object's initial rotation. Default is False.
        physics_update_step (int): The number of physics update steps to apply after randomization. Default is -1 (no update).
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

    if physics_update_step > 0:
        env.sim.update(step=physics_update_step)


def randomize_robot_eef_pose(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
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
        env_ids (torch.Tensor | None): The environment IDs to apply the randomization.
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
        robot.set_qpos(new_qpos, env_ids=env_ids, joint_ids=joint_ids, target=False)
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

    # simulate 1 steps to let the robot reach the target pose.
    env.sim.update(step=1)


def randomize_robot_qpos(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    qpos_range: tuple[list[float], list[float]] | None = None,
    relative_qpos: bool = True,
    joint_ids: List[int] | None = None,
) -> None:
    """Randomize the initial joint positions of a robot in the environment.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | None): The environment IDs to apply the randomization.
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

    robot.set_qpos(
        qpos=current_qpos, env_ids=env_ids, joint_ids=joint_ids, target=False
    )
    robot.set_qpos(qpos=current_qpos, env_ids=env_ids, joint_ids=joint_ids)
    env.sim.update(step=1)


def randomize_articulation_root_pose(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    position_range: tuple[list[float], list[float]] | None = None,
    rotation_range: tuple[list[float], list[float]] | None = None,
    relative_position: bool = True,
    relative_rotation: bool = False,
    physics_update_step: int = -1,
) -> None:
    """Randomize the root pose of an articulation in the environment.

    This function randomizes the position and/or rotation of an articulation's root link.
    The articulation's root is the base frame that all other links are attached to.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | None): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        position_range (tuple[list[float], list[float]] | None): The range for the position randomization.
            Format: [[x_min, y_min, z_min], [x_max, y_max, z_max]].
        rotation_range (tuple[list[float], list[float]] | None): The range for the rotation randomization.
            The rotation is represented as Euler angles (roll, pitch, yaw) in degrees.
        relative_position (bool): Whether to randomize the position relative to the articulation's
            initial position. Default is True.
        relative_rotation (bool): Whether to randomize the rotation relative to the articulation's
            initial rotation. Default is False.
        physics_update_step (int): The number of physics update steps to apply after randomization.
            Default is -1 (no update).

    .. note::
        This function is similar to :func:`randomize_rigid_object_pose` but operates on
        articulations (multi-link rigid body systems) rather than single rigid objects.
    """
    if entity_cfg.uid not in env.sim.get_articulation_uid_list():
        return

    articulation: Articulation = env.sim.get_articulation(entity_cfg.uid)

    # Get current root pose
    current_root_pose = articulation.get_local_pose()[env_ids]

    # Extract position and rotation from current pose
    init_pos = current_root_pose[:, :3]
    quat = current_root_pose[:, 3:7]  # (N, 4) quaternion
    # Convert quaternion to rotation matrix
    init_rot = matrix_from_quat(quat)

    # Generate random pose using the same logic as rigid_object_pose
    pose = get_random_pose(
        init_pos=init_pos,
        init_rot=init_rot,
        position_range=position_range,
        rotation_range=rotation_range,
        relative_position=relative_position,
        relative_rotation=relative_rotation,
    )

    articulation.set_local_pose(pose, env_ids=env_ids)
    articulation.clear_dynamics(env_ids=env_ids)

    if physics_update_step > 0:
        env.sim.update(step=physics_update_step)


def randomize_target_pose(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[list[float], list[float]],
    rotation_range: tuple[list[float], list[float]] | None = None,
    relative_position: bool = False,
    relative_rotation: bool = False,
    reference_entity_cfg: SceneEntityCfg | None = None,
    store_key: str = "target_pose",
) -> None:
    """Randomize a virtual target pose and store in env state.

    This function generates random target poses without requiring a physical object in the scene.
    The generated poses are stored as a public attribute in env for use by observations and rewards.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | None): The environment IDs to apply the randomization.
        position_range (tuple[list[float], list[float]]): The range for the position randomization.
        rotation_range (tuple[list[float], list[float]] | None): The range for the rotation randomization.
            The rotation is represented as Euler angles (roll, pitch, yaw) in degree.
        relative_position (bool): Whether to randomize the position relative to a reference entity. Default is False.
        relative_rotation (bool): Whether to randomize the rotation relative to a reference entity. Default is False.
        reference_entity_cfg (SceneEntityCfg | None): The reference entity for relative randomization.
            If None and relative mode is True, uses world origin.
        store_key (str): The key to store the target pose in env state. Default is "target_pose".
            The pose will be stored as a public attribute env.{store_key}.
    """
    num_instance = len(env_ids)

    # Get reference pose if needed
    if relative_position or relative_rotation:
        if reference_entity_cfg is not None:
            # Get reference entity pose
            ref_obj = env.sim.get_rigid_object(reference_entity_cfg.uid)
            if ref_obj is not None:
                ref_pose = ref_obj.get_local_pose(to_matrix=True)[env_ids]
                init_pos = ref_pose[:, :3, 3]
                init_rot = ref_pose[:, :3, :3]
            else:
                # Fallback to world origin
                init_pos = torch.zeros(num_instance, 3, device=env.device)
                init_rot = (
                    torch.eye(3, device=env.device)
                    .unsqueeze(0)
                    .repeat(num_instance, 1, 1)
                )
        else:
            # Use world origin as reference
            init_pos = torch.zeros(num_instance, 3, device=env.device)
            init_rot = (
                torch.eye(3, device=env.device).unsqueeze(0).repeat(num_instance, 1, 1)
            )
    else:
        # Absolute randomization, init values won't be used
        init_pos = torch.zeros(num_instance, 3, device=env.device)
        init_rot = (
            torch.eye(3, device=env.device).unsqueeze(0).repeat(num_instance, 1, 1)
        )

    # Generate random pose
    pose = get_random_pose(
        init_pos=init_pos,
        init_rot=init_rot,
        position_range=position_range,
        rotation_range=rotation_range,
        relative_position=relative_position,
        relative_rotation=relative_rotation,
    )

    if not hasattr(env, store_key):
        setattr(
            env,
            store_key,
            torch.zeros(env.num_envs, 4, 4, device=env.device, dtype=torch.float32),
        )

    target_poses = getattr(env, store_key)
    target_poses[env_ids] = pose


class planner_grid_cell_sampler(Functor):
    """Sample grid cells for object placement without replacement.

    This functor divides a planar region into a regular 2D grid and samples cells
    to place objects. Each sampled cell will be marked as occupied and will not be
    resampled until the grid is reset.

    The sampler places objects at the center of selected grid cells, with the z-position
    set to a reference height.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        """Initialize the GridCellSampler functor.

        Args:
            cfg: The configuration of the functor.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # Initialize the grid state (will be properly set in reset)
        self._grid_state: dict[int, torch.Tensor] = {}
        self._grid_cell_sizes: dict[int, tuple[float, float]] = {}

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the grid sampling state.

        Args:
            env_ids: The environment IDs to reset. If None, resets all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self._env.device)

        for env_id in env_ids:
            env_id_int = (
                int(env_id.item()) if isinstance(env_id, torch.Tensor) else int(env_id)
            )
            # Initialize grid as all zeros (all cells available)
            if env_id_int in self._grid_state:
                self._grid_state[env_id_int].fill_(0)

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: torch.Tensor | None,
        position_range: tuple[list[float], list[float]],
        reference_height: float,
        object_uid_list: list[str],
        grid_size: tuple[int, int],
        physics_update_step: int = -1,
    ) -> None:
        """Sample grid cells and place objects at those positions.

        Args:
            env: The environment instance.
            env_ids: The environment IDs to apply sampling. If None, applies to all environments.
            position_range: The planar range [(x_min, y_min), (x_max, y_max)] defining the region.
            reference_height: The z-coordinate for placing objects [m].
            object_uid_list: List of rigid object UIDs to place in the grid cells.
            grid_size: A tuple (rows, cols) defining the grid dimensions.
            physics_update_step: The number of physics update steps to apply after placement. Default is -1 (no update).

        Returns:
            None
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=env.device)

        self.reset(env_ids)
        num_instance = len(env_ids)

        # Parse position range
        x_min, y_min = position_range[0]
        x_max, y_max = position_range[1]
        cols, rows = grid_size

        obj_positions = []
        obj_list = []
        # Verify all objects exist
        for obj_uid in object_uid_list:
            if obj_uid not in env.sim.get_rigid_object_uid_list():
                logger.log_warning(
                    f"Object UID '{obj_uid}' not found in the simulation."
                )
                continue
            obj_positions.append(
                torch.zeros((num_instance, 3), dtype=torch.float32, device=env.device)
            )
            obj = env.sim.get_rigid_object(obj_uid)
            obj.reset()
            obj_list.append(obj)

        # Calculate cell dimensions
        cell_width = (x_max - x_min) / cols
        cell_height = (y_max - y_min) / rows

        # Initialize grid state for environments if not present
        for env_id in env_ids:
            env_id_int = (
                int(env_id.item()) if isinstance(env_id, torch.Tensor) else int(env_id)
            )
            if env_id_int not in self._grid_state:
                self._grid_state[env_id_int] = torch.zeros(
                    rows, cols, device=env.device, dtype=torch.uint8
                )
            self._grid_cell_sizes[env_id_int] = (cell_width, cell_height)

        # Batch operation: for each environment, place all objects
        for env_id in env_ids:
            env_id_int = (
                int(env_id.item()) if isinstance(env_id, torch.Tensor) else int(env_id)
            )
            grid = self._grid_state[env_id_int]

            # Sample and place each object in this environment
            for obj_id, rigid_object in enumerate(obj_list):
                # Find available cells
                available_cells = torch.where(grid == 0)

                if len(available_cells[0]) == 0:
                    logger.log_warning(
                        f"No available cells in grid for environment {env_id_int}. All cells occupied."
                    )
                    break

                # Randomly sample an available cell
                num_available = len(available_cells[0])
                random_idx = torch.randint(
                    0, num_available, (1,), device=env.device
                ).item()

                row = available_cells[0][random_idx].item()
                col = available_cells[1][random_idx].item()

                # Mark cell as occupied
                grid[row, col] = 1

                # Calculate position at cell center
                x = x_min + (col + 0.5) * cell_width
                y = y_min + (row + 0.5) * cell_height
                z = reference_height

                obj_positions[obj_id][env_id] = torch.tensor(
                    [x, y, z], dtype=torch.float32, device=env.device
                )

        for obj_id, rigid_object in enumerate(obj_list):
            rigid_object: RigidObject
            pose = rigid_object.get_local_pose()[env_ids]
            pose[:, 0:3] = obj_positions[obj_id]

            # Set object pose
            rigid_object.set_local_pose(pose, env_ids=env_ids)
            rigid_object.clear_dynamics()

        if physics_update_step > 0:
            env.sim.update(step=physics_update_step)


@configclass
class randomize_anchor_height_cfg(FunctorCfg):
    """Configuration for the randomize_anchor_height functor.

    This functor randomizes the Z height of an anchor object (e.g., a table) and
    applies the same height delta to all other included scene objects, preserving
    their XY position and rotation.
    """

    anchor_uid: str = MISSING
    """Exact UID of the anchor object whose height is randomized."""

    height_delta_range: tuple[list[float], list[float]] | None = None
    """Uniform sampling range for the height delta: ([z_min], [z_max])."""

    height_delta_candidates: list[float] | None = None
    """Discrete set of allowed height delta values."""

    include_groups: list[str] | None = None
    """Object groups to shift. ``None`` means all groups are included."""

    exclude_uids: list[str] = []
    """Additional UIDs to skip beyond the anchor object."""

    mode: str = "reset"
    """Event mode (``startup``, ``interval``, or ``reset``)."""

    physics_update_step: int = 0
    """Number of physics update steps to apply after moving objects."""

    store_key: str = "anchor_height_delta"
    """Attribute name on ``env`` where the sampled delta is stored."""


class randomize_anchor_height(Functor):
    """Randomize the height of an anchor object and shift other objects by the same delta.

    The functor samples a per-environment height delta, moves the anchor object
    relative to its configured initial position, and adds the same delta to the
    Z component of every other included object while preserving XY and rotation.
    """

    _VALID_GROUPS = {"background", "rigid_object", "rigid_object_group", "articulation"}

    def __init__(self, cfg: randomize_anchor_height_cfg, env: EmbodiedEnv):
        """Initialize the functor and resolve affected object UIDs.

        Args:
            cfg: The functor configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # Validate sampling configuration
        if cfg.height_delta_range is None and cfg.height_delta_candidates is None:
            raise ValueError(
                "Either 'height_delta_range' or 'height_delta_candidates' must be provided."
            )
        if (
            cfg.height_delta_candidates is not None
            and len(cfg.height_delta_candidates) == 0
        ):
            raise ValueError("'height_delta_candidates' must not be empty.")
        if (
            cfg.height_delta_range is not None
            and cfg.height_delta_candidates is not None
        ):
            logger.log_warning(
                "Both 'height_delta_range' and 'height_delta_candidates' provided; "
                "using 'height_delta_range'."
            )

        # Resolve include groups
        include_groups = cfg.include_groups
        if include_groups is None:
            include_groups = [
                "background",
                "rigid_object",
                "rigid_object_group",
                "articulation",
            ]
        invalid_groups = set(include_groups) - self._VALID_GROUPS
        if invalid_groups:
            raise ValueError(
                f"Invalid include_groups: {sorted(invalid_groups)}. "
                f"Valid options are: {sorted(self._VALID_GROUPS)}."
            )
        self._include_groups = include_groups

        # Confirm anchor exists
        anchor = self._get_object(cfg.anchor_uid)
        if anchor is None:
            raise ValueError(
                f"Anchor object with uid '{cfg.anchor_uid}' not found in the scene."
            )
        self._anchor = anchor

        # Build affected UID list
        self._affected_uids = self._resolve_affected_uids(
            env, cfg.anchor_uid, cfg.exclude_uids
        )

    def _get_object(self, uid: str):
        """Get a rigid object, articulation, or rigid object group by UID."""
        if uid in self._env.sim.get_rigid_object_uid_list():
            return self._env.sim.get_rigid_object(uid)
        if uid in self._env.sim.get_articulation_uid_list():
            return self._env.sim.get_articulation(uid)
        if (
            hasattr(self._env.sim, "get_rigid_object_group_uid_list")
            and uid in self._env.sim.get_rigid_object_group_uid_list()
        ):
            return self._env.sim.get_rigid_object_group(uid)
        return None

    def _resolve_affected_uids(
        self, env: EmbodiedEnv, anchor_uid: str, exclude_uids: list[str]
    ) -> list[str]:
        """Resolve the list of UIDs that should be shifted."""
        uids: set[str] = set()
        if any(g in self._include_groups for g in ("background", "rigid_object")):
            uids.update(env.sim.get_rigid_object_uid_list())
        if "rigid_object_group" in self._include_groups:
            if hasattr(env.sim, "get_rigid_object_group_uid_list"):
                uids.update(env.sim.get_rigid_object_group_uid_list())
        if "articulation" in self._include_groups:
            uids.update(env.sim.get_articulation_uid_list())

        exclude = set(exclude_uids) | {anchor_uid}
        return sorted(uids - exclude)

    def _sample_delta(self, num_envs: int) -> torch.Tensor:
        """Sample a height delta for each environment."""
        cfg = self.cfg
        device = self._env.device

        if cfg.height_delta_range is not None:
            low = torch.tensor(cfg.height_delta_range[0], device=device)
            high = torch.tensor(cfg.height_delta_range[1], device=device)
            return sample_uniform(
                lower=low, upper=high, size=(num_envs, 1), device=device
            ).squeeze(-1)

        # Discrete sampling
        candidates = torch.tensor(cfg.height_delta_candidates, device=device)
        indices = torch.randint(0, len(candidates), (num_envs,), device=device)
        return candidates[indices]

    def _move_object_z(
        self, obj, delta_z: torch.Tensor, env_ids: torch.Tensor, absolute: bool = False
    ) -> None:
        """Move an object in Z by delta_z.

        Args:
            obj: The object to move (RigidObject, Articulation, or RigidObjectGroup).
            delta_z: Per-environment Z offset.
            env_ids: Target environment IDs.
            absolute: If True, set Z to obj.cfg.init_pos[2] + delta_z.
                      If False, add delta_z to the current Z.
        """
        if isinstance(obj, RigidObjectGroup):
            # RigidObjectGroup does not have a single init_pos; always shift relative to current pose.
            pose = obj.get_local_pose(to_matrix=True)  # (N, M, 4, 4)
            pose[env_ids, :, 2, 3] += delta_z.unsqueeze(-1)
            obj.set_local_pose(pose[env_ids], env_ids=env_ids)
            return

        pose = obj.get_local_pose()
        if pose.ndim == 3:
            # (N, 4, 4) matrix form from RigidObject
            current_z = pose[env_ids, 2, 3]
            if absolute:
                init_z = torch.tensor(
                    obj.cfg.init_pos[2], dtype=torch.float32, device=obj.device
                )
                new_z = init_z + delta_z
            else:
                new_z = current_z + delta_z
            pose[env_ids, 2, 3] = new_z
        else:
            # (N, 7) vector form from Articulation: (x, y, z, qw, qx, qy, qz)
            current_z = pose[env_ids, 2]
            if absolute:
                init_z = torch.tensor(
                    obj.cfg.init_pos[2], dtype=torch.float32, device=obj.device
                )
                new_z = init_z + delta_z
            else:
                new_z = current_z + delta_z
            pose[env_ids, 2] = new_z

        obj.set_local_pose(pose[env_ids], env_ids=env_ids)
        if hasattr(obj, "clear_dynamics"):
            if isinstance(obj, Articulation):
                obj.clear_dynamics(env_ids=env_ids)
            else:
                obj.clear_dynamics()

    def __call__(self, env: EmbodiedEnv, env_ids: torch.Tensor | None) -> None:
        """Apply the height randomization.

        Args:
            env: The environment instance.
            env_ids: Target environment IDs. If None, all environments are targeted.
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)

        if len(env_ids) == 0:
            return

        num_envs = len(env_ids)
        delta_z = self._sample_delta(num_envs)

        # Move anchor relative to its initial pose
        self._move_object_z(self._anchor, delta_z, env_ids, absolute=True)

        # Move affected objects relative to their current pose
        for uid in self._affected_uids:
            obj = self._get_object(uid)
            if obj is None:
                logger.log_warning(
                    f"Affected object '{uid}' no longer exists; skipping height shift."
                )
                continue
            self._move_object_z(obj, delta_z, env_ids, absolute=False)

        # Physics settle
        if self.cfg.physics_update_step > 0:
            env.sim.update(step=self.cfg.physics_update_step)

        # Store delta for downstream use
        if self.cfg.store_key:
            setattr(env, self.cfg.store_key, delta_z)
