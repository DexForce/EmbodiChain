# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

from __future__ import annotations

import torch
import os
import random
import numpy as np
import traceback
from copy import deepcopy
from typing import TYPE_CHECKING, List, Union, Tuple, Dict, Set
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import gaussian_filter
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import dexsim
from embodichain.lab.sim.objects import (
    RigidObject,
    RigidObjectGroup,
)
from embodichain.utils.math import (
    sample_uniform,
)
from embodichain.lab.gym.envs.managers.randomization.spatial import get_random_pose
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.sim.cfg import RigidObjectCfg, RigidBodyAttributesCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.lab.gym.envs.managers.events import replace_assets_from_group
from embodichain.utils.file import get_all_files_in_directory
from dexechain.utils import logger
from dexechain.utils.utility_3d import compute_scale_and_rotation_to_match
from dexechain.data import get_data_path

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


class replace_assets_from_category(replace_assets_from_group):
    """Swap an asset with another asset picked from the same category folder."""

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the functor.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)
        self._base_body_scale = tuple(self.asset_cfg.body_scale)
        self._base_shape_fpath = self.asset_cfg.shape.fpath
        self._saved_pose = None
        ref_path = self._base_shape_fpath
        if not os.path.isabs(ref_path):
            ref_path = get_data_path(ref_path)
        self._reference_mesh_path = ref_path

        category = cfg.params.get("category")
        self._full_path = os.path.join(self._full_path, category)
        all_files = get_all_files_in_directory(self._full_path)
        mesh_types = (".glb", ".obj", ".stl", ".ply", ".gltf")

        self._asset_group_path = [
            f for f in all_files if f.lower().endswith(mesh_types)
        ]
        if not self._asset_group_path:
            logger.log_warning(
                f"No .ply mesh files found in category folder: {self._full_path}. Please check your assets."
            )

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        entity_cfg: SceneEntityCfg,
        folder_path: str,
        category: str = None,
        remove_prob: float = 0.0,
    ) -> None:

        asset = env.sim.get_asset(entity_cfg.uid)
        if asset is not None:
            self.visual_mats = asset.get_visual_material_inst()
            if env_ids is not None:
                self._saved_pose = asset.get_local_pose(to_matrix=False)[env_ids, :]
            else:
                self._saved_pose = asset.get_local_pose(to_matrix=False)
            env.sim.remove_asset(entity_cfg.uid)

        candidate_paths = self._asset_group_path.copy()
        random.shuffle(candidate_paths)

        for asset_path in candidate_paths:
            try:
                scale_xyz, scale_mean, _ = compute_scale_and_rotation_to_match(
                    ref_path=str(self._reference_mesh_path),
                    target_path=str(asset_path),
                )
                if scale_xyz.size == 0 or np.isclose(scale_xyz.min(), 0.0):
                    continue
                ratio = float(scale_xyz.max() / max(scale_xyz.min(), 1e-6))
                if ratio > 2.5:
                    continue

                self.asset_cfg.shape.fpath = asset_path
                base_scale = np.array(self._base_body_scale, dtype=np.float32)
                self.asset_cfg.body_scale = tuple((base_scale).tolist())

                break
            except Exception:
                continue
        else:
            self.asset_cfg.shape.fpath = self._base_shape_fpath
            self.asset_cfg.body_scale = tuple(self._base_body_scale)

        if self.asset_type == RigidObject:
            if random.random() >= remove_prob:
                asset = env.sim.add_rigid_object(cfg=self.asset_cfg)
                asset.share_visual_material_inst(self.visual_mats)
                asset.set_local_pose(self._saved_pose, env_ids=env_ids)
        else:
            logger.log_error("Only RigidObject assets are supported for replacement.")


def set_detached_uids_for_env_reset(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    uids: list[str],
) -> None:
    """Set the UIDs of objects that are detached from automatic reset in the environment.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | None): The environment IDs to apply the event.
        uids (list[str]): The list of UIDs to be detached from automatic reset.
    """

    env.add_detached_uids_for_reset(uids=uids)


def drop_rigid_object_group_sequentially_once(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    drop_position: List[float] = [0.0, 0.0, 1.0],
    position_range: Tuple[List[float], List[float]] = (
        [-0.1, -0.1, 0.0],
        [0.1, 0.1, 0.0],
    ),
    physics_step: int = 2,
) -> None:
    """
    Drop all objects in a rigid object group from a specified height sequentially
    and cache their initial positions for future resets. Only call this at environment creation.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | None): The environment IDs to apply the event.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        drop_position (List[float]): The base position from which to drop the objects.
        position_range (Tuple[List[float], List[float]]): Range to randomize the drop position.
        physics_step (int): Number of physics steps to simulate after dropping.
    """
    obj_group: RigidObjectGroup = env.sim.get_rigid_object_group(entity_cfg.uid)

    if obj_group is None:
        logger.log_error(
            f"RigidObjectGroup with UID '{entity_cfg.uid}' not found in the simulation."
        )

    num_instance = len(env_ids)
    num_objects = obj_group.num_objects

    range_low = torch.tensor(position_range[0], device=env.device)
    range_high = torch.tensor(position_range[1], device=env.device)
    drop_pos = (
        torch.tensor(drop_position, device=env.device)
        .unsqueeze_(0)
        .repeat(num_instance, 1)
    )
    drop_pose = torch.zeros((num_instance, 7), device=env.device)
    drop_pose[:, 3] = 1.0  # w component of quaternion
    drop_pose[:, :3] = drop_pos
    for i in range(num_objects):
        random_offset = sample_uniform(
            lower=range_low,
            upper=range_high,
            size=(num_instance, 3),
            device=env.device,
        )
        drop_pose_i = drop_pose.unsqueeze(1)
        drop_pose_i[:, 0, :3] = drop_pos + random_offset

        obj_group.set_local_pose(pose=drop_pose_i, env_ids=env_ids, obj_ids=[i])

        env.sim.update(step=physics_step)

    env.sim.update(step=500)
    # Cache for reset/recovery, key is entity_cfg.uid or cache_key if given
    if not hasattr(env, "_rigidobjectgroup_init_poses"):
        env._rigidobjectgroup_init_poses = {}
    env._rigidobjectgroup_init_poses[entity_cfg.uid] = (
        obj_group.get_local_pose().detach().cpu().numpy()
    )


def restore_changed_rigid_object_group_objects(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    drop_position: List[float] = [0.0, 0.0, 1.0],
    position_range: Tuple[List[float], List[float]] = (
        [-0.1, -0.1, 0.0],
        [0.1, 0.1, 0.0],
    ),
    physics_step: int = 2,
    changed_obj_ids: list[int] = None,
) -> None:
    """
    Restore only those objects in a rigid object group that have changed position.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | None): The environment IDs to apply the event.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        drop_position (List[float]): The base position from which to drop the objects.
        position_range (Tuple[List[float], List[float]]): Range to randomize the drop position.
        physics_step (int): Number of physics steps to simulate after dropping.
        changed_obj_ids (list[int]): Indices of objects that drifted and should be restored.
    """

    obj_group: RigidObjectGroup = env.sim.get_rigid_object_group(entity_cfg.uid)
    if obj_group is None:
        logger.log_error(
            f"RigidObjectGroup with UID '{entity_cfg.uid}' not found in the simulation."
        )

    num_objects = obj_group.num_objects
    # Retrieve cached poses
    cache = getattr(env, "_rigidobjectgroup_init_poses", {})
    init_poses = cache.get(entity_cfg.uid, None)
    if init_poses is None:
        logger.log_warning(f"No cached initial poses found for UID '{entity_cfg.uid}'.")
        drop_rigid_object_group_sequentially_once(
            env=env,
            env_ids=env_ids,
            entity_cfg=entity_cfg,
            drop_position=drop_position,
            position_range=position_range,
            physics_step=physics_step,
        )
        return
    if changed_obj_ids is None:
        changed_obj_ids = list(range(num_objects))
    for obj_id in changed_obj_ids:
        pose = torch.from_numpy(init_poses[:, obj_id]).to(
            env.device
        )  # (num_instance,7)
        obj_group.set_local_pose(
            pose=pose.unsqueeze(1), env_ids=env_ids, obj_ids=[obj_id]
        )
    env.sim.update(step=100)


def _validate_affordance_type(affordance_type: str) -> bool:
    """校验 affordance_type 是否已注册在 ObjectAffordanceType 中。"""
    from dexechain.data.enum import ObjectAffordanceType

    valid_values = {e.value for e in ObjectAffordanceType}
    if affordance_type not in valid_values:
        logger.log_error(
            f"affordance_type '{affordance_type}' is not a valid ObjectAffordanceType. "
            f"Allowed values: {sorted(valid_values)}"
        )
        return False
    return True


def reset_affordance_occupancy(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    affordance_type: str = "place",
) -> None:
    """
    Reset the affordance occupancy for a specific type on an object.
    Only 'common' eef_type points are tracked by affordance_occupancy.

    Args:
        affordance_type: The affordance type to reset, must be a valid ObjectAffordanceType value.
    """
    if not _validate_affordance_type(affordance_type):
        return

    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return
    ref_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)
    if ref_object is None:
        logger.log_error(
            f"RigidObject with UID '{entity_cfg.uid}' not found in the simulation."
        )
        return

    pts = ref_object.affordance_points_object
    if pts is None or not isinstance(pts, dict):
        ref_object.affordance_occupancy = None
        return

    # affordance_occupancy is {type: tensor(N, bool)}
    occupancy = getattr(ref_object, "affordance_occupancy", None)
    if not isinstance(occupancy, dict):
        occupancy = {}

    if affordance_type in pts and "common" in pts[affordance_type]:
        n = pts[affordance_type]["common"].shape[0]
        occupancy[affordance_type] = torch.zeros(n, dtype=torch.bool, device=env.device)
    else:
        occupancy.pop(affordance_type, None)

    ref_object.affordance_occupancy = occupancy if occupancy else None


def set_object_pose_by_affordance_points(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    ref_object_uid: str,
    affordance_type: str = "place",
    affordance_indices: List[int] = None,
    position_range: tuple[list[float], list[float]] | None = None,
    rotation_range: tuple[list[float], list[float]] | None = None,
    relative_position: bool = True,
    relative_rotation: bool = False,
) -> None:
    """
    Set the pose of an object using the 'common' affordance points of a reference object.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply the event.
        entity_cfg: The configuration of the scene entity to randomize.
        ref_object_uid: The UID of the reference object.
        affordance_type: The affordance type to use, must be a valid ObjectAffordanceType value.
        affordance_indices: Optional list of specific point indices to choose from.
        position_range: Optional position randomization range.
        rotation_range: Optional rotation randomization range.
        relative_position: Whether position_range is relative.
        relative_rotation: Whether rotation_range is relative.
    """
    if not _validate_affordance_type(affordance_type):
        return

    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return

    rigid_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)

    if (
        hasattr(rigid_object, "occurrence_probability")
        and random.random() > rigid_object.occurrence_probability
    ):
        rigid_object.set_local_pose(
            torch.eye(4, dtype=torch.float32, device=env.device).unsqueeze(0),
            env_ids=env_ids,
        )
        return

    ref_object: RigidObject = env.sim.get_rigid_object(ref_object_uid)
    if ref_object is None:
        logger.log_error(
            f"RigidObject with UID '{ref_object_uid}' not found in the simulation."
        )
        return
    ref_object_pose = ref_object.get_local_pose(to_matrix=True)

    # Get affordance points dict: {type: {eef_type: tensor(N,4,4)}}
    pts_dict = getattr(ref_object, "affordance_points_object", None)
    if pts_dict is None or not isinstance(pts_dict, dict):
        logger.log_error(
            f"Reference object '{ref_object_uid}' does not have affordance_points_object defined."
        )
        return

    if affordance_type not in pts_dict:
        logger.log_error(
            f"Reference object '{ref_object_uid}' has no affordance type '{affordance_type}'."
        )
        return

    eef_dict = pts_dict[affordance_type]
    if "common" not in eef_dict:
        logger.log_error(
            f"Reference object '{ref_object_uid}' type '{affordance_type}' has no 'common' eef_type affordance points."
        )
        return

    affordance_points = eef_dict["common"]  # tensor(N, 4, 4)
    if affordance_points.shape[0] == 0:
        logger.log_error(
            f"Reference object '{ref_object_uid}' type '{affordance_type}' has empty 'common' affordance points."
        )
        return

    # Make sure reference object pose shape is (num_envs, 4, 4)
    if ref_object_pose.dim() == 2:
        ref_object_pose = ref_object_pose.unsqueeze(0)
    num_envs = ref_object_pose.shape[0]

    # Prefer affordance points that are not occupied
    occupancy = getattr(ref_object, "affordance_occupancy", None)
    type_occupancy = None
    if isinstance(occupancy, dict):
        type_occupancy = occupancy.get(affordance_type)

    if type_occupancy is not None:
        all_available_indices = (~type_occupancy).nonzero(as_tuple=True)[0].tolist()
    else:
        logger.log_warning(
            f"Reference object '{ref_object_uid}' does not track affordance occupancy for type '{affordance_type}'. "
            f"All affordance points are considered available."
        )
        all_available_indices = list(range(affordance_points.shape[0]))

    # If affordance_indices argument is provided, take intersection with available indices
    if affordance_indices is not None:
        affordance_indices_set = set(int(idx) for idx in affordance_indices)
        available_indices = list(affordance_indices_set & set(all_available_indices))
    else:
        available_indices = all_available_indices

    if not available_indices:
        logger.log_warning(
            f"No available affordance points to place object '{entity_cfg.uid}' "
            f"(type='{affordance_type}'); placement failed and object removed."
        )
        if entity_cfg.uid in env.sim.get_rigid_object_uid_list():
            rigid_object.set_local_pose(
                torch.eye(4, dtype=torch.float32, device=env.device).unsqueeze(0),
                env_ids=env_ids,
            )
        return

    chosen_index = random.choice(available_indices)
    # Mark the chosen affordance point as occupied
    if type_occupancy is not None:
        type_occupancy[chosen_index] = True

    # Compute the world coordinate of the selected point: (num_envs, 4, 4)
    world_affordance_points = []
    for env_idx in range(num_envs):
        world_pose = torch.matmul(
            ref_object_pose[env_idx], affordance_points[chosen_index]
        )
        world_affordance_points.append(world_pose)
    world_affordance_points = torch.stack(world_affordance_points, dim=0)

    world_affordance_points = get_random_pose(
        init_pos=world_affordance_points[:, :3, 3],
        init_rot=world_affordance_points[:, :3, :3],
        position_range=position_range,
        rotation_range=rotation_range,
        relative_position=relative_position,
        relative_rotation=relative_rotation,
    )

    rigid_object.set_local_pose(world_affordance_points, env_ids=env_ids)
    rigid_object.clear_dynamics()


def temporary_modify_physical_properties(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    temp_physics: dict = None,
    update_step: int = 100,
) -> None:
    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return
    rigid_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)
    if rigid_object is None:
        logger.log_error(
            f"RigidObject with UID '{entity_cfg.uid}' not found in the simulation."
        )
        return
    # import pdb; pdb.set_trace()
    original_physics_attrs = deepcopy(rigid_object.cfg.attrs)

    # 如果未传入临时物理属性，则使用默认临时值
    if not temp_physics:
        tmp_physics_attrs = RigidBodyAttributesCfg(
            mass=1.0, dynamic_friction=0.1, static_friction=0.1, restitution=0.1
        )
    else:
        # 基于当前物理属性生成临时属性，再更新对应key
        tmp_physics_attrs = deepcopy(original_physics_attrs)
        for key, value in temp_physics.items():
            if hasattr(tmp_physics_attrs, key):
                setattr(tmp_physics_attrs, key, value)
            else:
                logger.log_warning(f"RigidBodyAttributesCfg has no attribute '{key}'")

    rigid_object.set_attrs(tmp_physics_attrs, env_ids=env_ids)
    rigid_object.clear_dynamics()
    env.sim.update(step=update_step)
    rigid_object.set_attrs(original_physics_attrs, env_ids=env_ids)
    rigid_object.clear_dynamics()
    env.sim.update(step=update_step)


def move_rigid_object_randomly(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    position_range: tuple[list[float], list[float]] | None = None,
    rotation_range: tuple[list[float], list[float]] | None = None,
    physics_update_step: int = -1,
) -> None:
    """
    Randomly move the rigid object relative to its current pose.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | None): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        position_range (tuple[list[float], list[float]] | None): Position randomization range.
        rotation_range (tuple[list[float], list[float]] | None): Rotation (Euler degrees) randomization range.
        physics_update_step (int): Number of physics steps to update after the move.
    """
    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return

    rigid_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)

    current_pose = rigid_object.get_local_pose(to_matrix=True)
    if current_pose.dim() == 2:
        current_pose = current_pose.unsqueeze(0)
    init_pos = current_pose[:, :3, 3]
    init_rot = current_pose[:, :3, :3]

    pose = get_random_pose(
        init_pos=init_pos,
        init_rot=init_rot,
        position_range=position_range,
        rotation_range=rotation_range,
        relative_position=True,
        relative_rotation=True,
    )

    rigid_object.set_local_pose(pose, env_ids=env_ids)
    rigid_object.clear_dynamics()

    if physics_update_step > 0:
        env.sim.update(step=physics_update_step)


def export_scene_usd_once(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    usd_path: str = "scene.usda",
    stabilize_steps: int = 1000,
    overwrite: bool = False,
) -> None:
    """Export the current simulation scene to a USD file once after stabilization.

    This event is intended to be registered in the env config and only enabled
    when needed (e.g., for real2sim preview / debugging). It will:

    1) Optionally run several physics steps to let the scene settle.
    2) Export the whole scene to a USD file via ``sim.export_usd``.
    3) Ensure the export happens only once per env instance.

    Args:
        env: The environment instance.
        env_ids: Environment IDs (unused; kept for Event signature compatibility).
        usd_path: Target USD file path. Can be absolute or relative.
        stabilize_steps: Number of physics steps to run before export.
        overwrite: Whether to overwrite an existing USD file.
    """
    from pathlib import Path

    # Guard: only export once per env instance.
    if getattr(env, "_scene_usd_exported", False):
        return

    sim = env.sim

    # Let the scene stabilize if requested.
    if stabilize_steps > 0:
        sim.update(step=stabilize_steps)

    path = Path(usd_path)
    if not path.is_absolute():
        # Use current working directory as base for relative paths.
        path = Path.cwd() / path

    if path.exists() and not overwrite:
        # Mark as exported to avoid repeated checks in later calls.
        env._scene_usd_exported = True
        logger.log_warning(
            f"USD export skipped because file already exists: {str(path)}"
        )
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        sim.export_usd(str(path))
        env._scene_usd_exported = True
        logger.log_info(f"Scene USD exported to: {str(path)}")
    except Exception as e:
        logger.log_warning(f"Failed to export scene USD: {e}")


def adjust_object_center_of_mass(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    com_pos_offset: list[float] | torch.Tensor,
    physics_update_step: int = -1,
) -> None:
    """
    Adjust the center of mass (COM) position of the specified rigid body object.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | list[int]): The environment IDs to adjust.
        entity_cfg (SceneEntityCfg): The target entity configuration.
        com_pos_offset (list[float] | torch.Tensor): Desired COM offset (xyz coordinates, unit: meter), set absolutely as default_com_pose + offset.
        physics_update_step (int): Physics steps after adjustment; if >0, sim.update is called immediately.
    """
    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return

    rigid_object: "RigidObject" = env.sim.get_rigid_object(entity_cfg.uid)
    if getattr(rigid_object, "is_non_dynamic", False):
        logger.log_warning(
            f"Cannot adjust center of mass for non-dynamic rigid object '{entity_cfg.uid}'."
        )
        return

    # env_ids can be list[int] or torch.Tensor
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(list(env_ids), device=env.device)
    num_instance = len(env_ids)

    # com_pos_offset can be 1D or 2D (to support batch)
    if isinstance(com_pos_offset, list):
        com_pos_offset = torch.tensor(com_pos_offset, dtype=torch.float32)
    if com_pos_offset.ndim == 1:
        com_pos_offset = com_pos_offset.unsqueeze(0).repeat(num_instance, 1)
    elif com_pos_offset.shape[0] != num_instance:
        com_pos_offset = com_pos_offset.repeat(num_instance, 1)

    # Get the default com pose
    com = rigid_object.body_data.default_com_pose[env_ids]
    updated_com = com.clone()
    updated_com[:, 0:3] = com[:, 0:3] + com_pos_offset

    rigid_object.set_com_pose(updated_com, env_ids=env_ids)
    rigid_object.clear_dynamics()

    if physics_update_step > 0:
        env.sim.update(step=physics_update_step)


class dexgen_randomize_scene(Functor):
    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        super().__init__(cfg, env)
        params = cfg.params or {}
        self.cached_objects = deepcopy(params.get("cached_objects", []))
        self.random_percent = float(params.get("random_percent", 1.0))
        self.random_percent = max(0.0, min(1.0, self.random_percent))
        self._loaded_safe_uids: Set[str] = set()

    def _build_rigid_cfg(self, obj_dict: Dict) -> RigidObjectCfg:
        uid = obj_dict.get("uid")
        shape_cfg = obj_dict.get("shape", {})
        fpath = shape_cfg.get("fpath", None)
        mesh = MeshCfg(
            fpath=fpath,
            compute_uv=shape_cfg.get("compute_uv", False),
            shape_type=shape_cfg.get("shape_type", "Mesh"),
        )
        body_scale = tuple(obj_dict.get("body_scale", [1.0, 1.0, 1.0]))
        init_local_pose = obj_dict.get("init_local_pose")
        attrs_dict = obj_dict.get("attrs", {})
        attrs = RigidBodyAttributesCfg.from_dict(attrs_dict)
        max_convex = obj_dict.get("max_convex_hull_num", 8)
        cfg = RigidObjectCfg(
            uid=uid,
            shape=mesh,
            body_scale=body_scale,
            init_local_pose=init_local_pose,
            attrs=attrs,
            body_type="dynamic",
            max_convex_hull_num=max_convex,
        )
        return cfg

    def _prepare_initial_state(
        self, env: EmbodiedEnv, env_ids: Union[torch.Tensor, None]
    ):
        interact_objs_info = []
        robot = env.sim.get_asset("dexforce_w1")
        far_away_pose = torch.tensor(
            [[1.0, 2.0, 100.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        robot.set_local_pose(far_away_pose)

        for uid in env.sim.get_rigid_object_uid_list():
            if uid.startswith("interact_"):
                original_pose = env.sim.get_asset(uid).get_local_pose()
                aabb = env.sim.get_rigid_object(uid=uid)._entities[0].get_aabb_attr()
                interact_objs_info.append(
                    {
                        "uid": uid,
                        "original_pose": (original_pose.clone()),
                        "aabb": aabb,
                    }
                )
                far_away_pose[0][1] = far_away_pose[0][1] + torch.randint(
                    10, 100, size=()
                )
                far_away_pose[0][0] = far_away_pose[0][0] + torch.randint(
                    10, 100, size=()
                )
                env.sim.get_asset(uid).set_local_pose(far_away_pose)

        table_scale = env.sim.get_rigid_object(uid="table")._entities[0].get_aabb_attr()
        return interact_objs_info, table_scale

    def _compute_table_avalibale_area(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        interact_objs_info: List[Dict],
    ):
        height_eps = 0.05
        compute_cam = env.sim._world.get_env().create_camera(
            "compute_cam", 1080, 1080, headless=True
        )
        compute_cam.look_at(
            location=[0.0, 0.0, 3.0],
            target=[0.0, 0.0, 0.0],
            up_vector=[1.0, 0.0, 0.0],
        )
        compute_cam.open_camera()
        compute_cam.render()
        position = compute_cam.get_position_map()

        xyz = position[..., :3]
        z = xyz[..., 2]
        valid_mask = np.abs(z) > 0.01
        xyz_valid = xyz[valid_mask]
        if xyz_valid.shape[0] == 0:
            return None
        heights = xyz_valid[:, 2]
        order = np.argsort(heights)
        sorted_heights = heights[order]
        sorted_points = xyz_valid[order]
        groups = []
        current = {"height": sorted_heights[0], "points": [sorted_points[0]]}
        for h, p in zip(sorted_heights[1:], sorted_points[1:]):
            if abs(h - current["height"]) <= height_eps:
                current["points"].append(p)
            else:
                groups.append(current)
                current = {"height": h, "points": [p]}
        groups.append(current)
        table_group = max(groups, key=lambda g: len(g["points"]))
        points = np.array(table_group["points"])
        xs, ys = points[:, 0], points[:, 1]
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        cell_size = 0.02
        sigma = 1.0
        threshold = 0.3
        nx = int(np.ceil((xmax - xmin) / cell_size))
        ny = int(np.ceil((ymax - ymin) / cell_size))
        grid = np.zeros((nx, ny), dtype=np.float32)
        ix = ((xs - xmin) / cell_size).astype(int)
        iy = ((ys - ymin) / cell_size).astype(int)
        for x, y in zip(ix, iy):
            if 0 <= x < nx and 0 <= y < ny:
                grid[x, y] += 1
        grid_prob = grid / grid.max()
        grid_smooth = gaussian_filter(grid_prob, sigma=sigma)
        occupancy = grid_smooth >= threshold  # True=桌面可用区域，False=不可用
        for obj_info in interact_objs_info:
            uid = obj_info["uid"]
            aabb = obj_info["aabb"]
            # [xmin, ymin, zmin, xmax, ymax, zmax]
            obj_xmin, obj_xmax = aabb[0], aabb[3]
            obj_ymin, obj_ymax = aabb[1], aabb[4]
            gx1 = int(np.floor((obj_xmin - xmin) / cell_size)) - 1
            gx2 = int(np.ceil((obj_xmax - xmin) / cell_size)) + 1
            gy1 = int(np.floor((obj_ymin - ymin) / cell_size)) - 1
            gy2 = int(np.ceil((obj_ymax - ymin) / cell_size)) + 1
            gx1 = max(0, gx1)
            gx2 = min(nx, gx2)
            gy1 = max(0, gy1)
            gy2 = min(ny, gy2)
            occupancy[gx1:gx2, gy1:gy2] = False

        # # 保存mask后的可视化图
        plt.figure()
        plt.imshow(occupancy.T, cmap="gray", origin="lower")
        plt.xlabel("X ")
        plt.ylabel("Y ")
        plt.title("Table")
        plt.savefig("table_occupancy_masked.png", bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"桌面高度 (z_world): {table_group['height']:.3f} m")
        print(f"X范围: [{xmin:.3f}, {xmax:.3f}]  (宽度 = {xmax - xmin:.3f} m)")
        print(f"Y范围: [{ymin:.3f}, {ymax:.3f}]  (深度 = {ymax - ymin:.3f} m)")
        print(f"网格形状: {occupancy.shape}, 网格大小: {cell_size} m")
        print(f"桌面可用面积: {np.sum(occupancy) * (cell_size ** 2):.3f} 平方米")

        # 清理相机
        env.sim._world.get_env().remove_camera("compute_cam")
        return {
            "z_world": float(table_group["height"]),
            "occupancy": occupancy,
            "cell_size": cell_size,
            "x_range": (float(xmin), float(xmax)),
            "y_range": (float(ymin), float(ymax)),
            "num_points": len(points),
        }

    def _compute_background_objects_position(
        self, env: EmbodiedEnv, env_ids: Union[torch.Tensor, None], table_info, selected
    ):
        # # debug only
        matplotlib.use("Agg")
        os.makedirs("debug_occupancy", exist_ok=True)
        visualize_every = 100
        def _save_occupancy_snapshot(
            step, curr_pos, footprint_offsets_list, grid_bool, grid_x_len, grid_y_len
        ):
            occ_snap = grid_bool.copy()
            pos_int_np = torch.round(curr_pos.detach().cpu()).numpy().astype(int)
            for i in range(pos_int_np.shape[0]):
                center = pos_int_np[i]
                offsets_t = footprint_offsets_list[i].detach().cpu().numpy().astype(int)
                world = offsets_t + center
                x_idx = world[:, 0]
                y_idx = world[:, 1]
                valid_mask = (
                    (x_idx >= 0)
                    & (x_idx < grid_x_len)
                    & (y_idx >= 0)
                    & (y_idx < grid_y_len)
                )
                if not valid_mask.any():
                    continue
                xi = x_idx[valid_mask]
                yi = y_idx[valid_mask]
                occ_snap[xi, yi] = False
                plt.figure(figsize=(6, 6))
                plt.imshow(occ_snap.T, cmap="gray", origin="lower")
                plt.title(f"Occupancy snapshot step {step}")
                plt.xlabel("X (grid)")
                plt.ylabel("Y (grid)")
                fn = os.path.join("debug_occupancy", f"occupancy_step_{step:05d}.png")
                plt.savefig(fn, bbox_inches="tight", pad_inches=0)
                plt.close()
        # # done

        sim = env.sim
        if table_info is None or not selected:
            return None

        grid_bool = table_info["occupancy"]
        cell_size = float(table_info["cell_size"])
        x_min, x_max = table_info["x_range"]
        y_min, y_max = table_info["y_range"]
        z_table = float(table_info["z_world"])
        grid_x_len, grid_y_len = grid_bool.shape
        logger.log_info(
            f"[DBG] Grid shape (nx,ny) = {grid_x_len, grid_y_len}, cell size = {cell_size}m"
        )
        if grid_x_len <= 0 or grid_y_len <= 0:
            logger.log_error("Failed, empty grid")
            return {"status": "fail", "reason": "empty grid"}

        class RoundSTE(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.round()

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output

        round_ste = RoundSTE.apply

        def aabb_from_raw(raw_aabb):
            xmin, xmax = raw_aabb[0], raw_aabb[3]
            ymin, ymax = raw_aabb[1], raw_aabb[4]
            return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

        def intersection_area(b1, b2):
            xmin = max(b1[0], b2[0])
            ymin = max(b1[1], b2[1])
            xmax = min(b1[2], b2[2])
            ymax = min(b1[3], b2[3])
            if xmax <= xmin or ymax <= ymin:
                return 0.0
            return (xmax - xmin) * (ymax - ymin)

        def build_object_data(self, o, sim, env_ids):
            uid = o.get("uid")
            if not uid:
                return None

            rigid_cfg = self._build_rigid_cfg(o)
            rigid_object = sim.add_rigid_object(cfg=rigid_cfg)

            local_pose = torch.tensor(
                rigid_cfg.init_local_pose, dtype=torch.float32
            ).unsqueeze(0)
            rigid_object.set_local_pose(local_pose, env_ids)

            raw_aabb = rigid_object._entities[0].get_aabb_attr()
            bbox = aabb_from_raw(raw_aabb)

            # raw_aabb: [xmin, ymin, zmin, xmax, ymax, zmax]
            # We'll use z-height for stack recovery.
            zmin = float(raw_aabb[2])
            zmax = float(raw_aabb[5])
            height_z = max(0.0, zmax - zmin)

            init_pose = np.array(o.get("init_local_pose"))
            pos = init_pose[:3, 3]
            height = float(pos[2])
            rigid_object.enable_collision(torch.tensor([0]))
            rigid_object._entities[0].enable_gravity(False)
            # rigid_object._entities[0].get_physical_body().set_physic_flag(dexsim.types.BodyFlag.KINEMATIC, True)
            rigid_object._entities[0].set_visible(False)
            size = bbox[2:] - bbox[:2]

            return {
                "uid": uid,
                "bbox": bbox,
                "height_z": float(height_z),
                "size": size,
                "half_size": size / 2.0,
                "pos": pos[:2].copy(),
                "init_pos_copy": pos[:2].copy(),
                "init_pose": init_pose,
                "height": height,
                "locked": False,
                "is_group": False,
            }

        def create_overlay_groups(objs):
            overlays = [o for o in objs if o["height"] > 0.01]
            bases = [o for o in objs if o["height"] <= 0.01]

            new_groups = []
            for overlay in overlays:
                overlay_area = np.prod(overlay["size"])

                candidates = [
                    b
                    for b in bases
                    if intersection_area(overlay["bbox"], b["bbox"])
                    / (overlay_area + 1e-6)
                    >= 0.7
                ]

                if not candidates:
                    overlay["height"] = 0.0
                    continue

                # Pick one base for stack height recovery (highest XY overlap).
                base = max(
                    candidates,
                    key=lambda b: intersection_area(overlay["bbox"], b["bbox"]),
                )
                overlay["stack_base_uid"] = base.get("uid")
                overlay["stack_base_height_z"] = float(base.get("height_z", 0.0))

                members = [overlay] + candidates

                bboxes = np.array([m["bbox"] for m in members])
                xmin, ymin = bboxes[:, :2].min(axis=0)
                xmax, ymax = bboxes[:, 2:].max(axis=0)

                union_bbox = np.array([xmin, ymin, xmax, ymax])
                union_center = (union_bbox[:2] + union_bbox[2:]) / 2.0
                new_uid = "_".join(sorted(m["uid"] for m in members))

                for m in members:
                    m["offset_from_group_center"] = m["pos"] - union_center
                    m["locked"] = True
                    m["group_id"] = new_uid

                new_groups.append(
                    {
                        "uid": new_uid,
                        "is_group": True,
                        "members": members,
                        "bbox": union_bbox,
                        "size": union_bbox[2:] - union_bbox[:2],
                        "half_size": (union_bbox[2:] - union_bbox[:2]) / 2.0,
                        "pos": union_center,
                        "init_pos_copy": union_center.copy(),
                        "height": 0.0,
                        "locked": False,
                    }
                )

                print(f"[DBG] Created overlay group: {new_uid}")

            return objs + new_groups

        def world_to_grid(pos, x_min, y_min, cell_size, gx_len, gy_len):
            gx = np.clip((pos[0] - x_min) / cell_size, 0, gx_len - 1)
            gy = np.clip((pos[1] - y_min) / cell_size, 0, gy_len - 1)
            return float(gx), float(gy)

        objs_data = []
        for o in selected:
            try:
                obj = build_object_data(self, o, sim, env_ids)
                if obj:
                    objs_data.append(obj)
            except Exception as e:
                logger.log_warning(f"[WARN] {o.get('uid')} failed: {e}")

        objs_data = create_overlay_groups(objs_data)
        opt_units = [o for o in objs_data if not o["locked"]]
        if not opt_units:
            logger.log_info("No units to optimize")
            return {"status": "ok", "placements": {}, "debug": {"msg": "no_units"}}

        logger.log_info(f"Number of units to optimize: {len(opt_units)}")

        init_pos_list = []
        sizes_in_cells = []
        uid_to_init_pose = {}

        for u in opt_units:
            gx, gy = world_to_grid(
                u["pos"], x_min, y_min, cell_size, grid_x_len, grid_y_len
            )
            init_pos_list.append([gx, gy])
            sx = max(1, int(np.ceil(u["size"][0] / cell_size)))
            sy = max(1, int(np.ceil(u["size"][1] / cell_size)))
            sizes_in_cells.append((sx, sy))
            uid_to_init_pose[u["uid"]] = u.get("init_pose")

        init_pos = torch.tensor(init_pos_list, dtype=torch.float32, requires_grad=False)
        curr_pos = init_pos.clone().detach().requires_grad_(True)
        opt_units = [obj for obj in objs_data if not obj.get("locked", False)]
        num_units = len(opt_units)

        def get_relative_offsets(w, h, margin, device=None, dtype=torch.float32):
            x0 = w // 2
            y0 = h // 2
            x = torch.arange(w + margin, dtype=dtype, device=device) - x0
            y = torch.arange(h + margin, dtype=dtype, device=device) - y0
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            offsets = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
            return offsets

        footprint_offsets_list = [
            get_relative_offsets(w, h, 1, device=curr_pos.device)
            for (w, h) in sizes_in_cells
        ]

        def _compute_dynamic_sdf(
            curr_pos,
            footprint_offsets_list,
            base_grid_bool,
            grid_x_len,
            grid_y_len,
            exclude_idx=None,
        ):
            dynamic_grid = base_grid_bool.copy()
            pos_int_np = torch.round(curr_pos.detach().cpu()).numpy().astype(int)
            for i in range(pos_int_np.shape[0]):
                if exclude_idx is not None and i == exclude_idx:
                    continue
                center = pos_int_np[i]
                offsets_t = footprint_offsets_list[i].detach().cpu().numpy().astype(int)
                world = offsets_t + center
                x_idx = world[:, 0]
                y_idx = world[:, 1]
                valid_mask = (
                    (x_idx >= 0)
                    & (x_idx < grid_x_len)
                    & (y_idx >= 0)
                    & (y_idx < grid_y_len)
                )
                if not valid_mask.any():
                    continue
                xi = x_idx[valid_mask]
                yi = y_idx[valid_mask]
                dynamic_grid[xi, yi] = False
            obstacle = (1 - dynamic_grid).astype(np.uint8)
            dt_obstacle_inside = distance_transform_edt(obstacle)
            dt_free_inside = distance_transform_edt(1 - obstacle)
            sdf_np = dt_free_inside.astype(np.float32) - dt_obstacle_inside.astype(
                np.float32
            )
            sdf_torch = (
                torch.from_numpy(sdf_np.T)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .contiguous()
            )
            return sdf_torch, dynamic_grid

        optimizer = torch.optim.Adam([curr_pos], lr=0.02)
        steps = 2500
        Wm1 = max(1, grid_x_len - 1)
        Hm1 = max(1, grid_y_len - 1)

        for step in range(steps):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=curr_pos.device)
            all_inside_counts = []

            for i in range(num_units):
                sdf_torch, _ = _compute_dynamic_sdf(
                    curr_pos,
                    footprint_offsets_list,
                    grid_bool,
                    grid_x_len,
                    grid_y_len,
                    exclude_idx=i,
                )
                pos_int = round_ste(curr_pos[i : i + 1])
                offsets = footprint_offsets_list[i].to(curr_pos.device)
                world = pos_int[0] + offsets

                x_coords = world[:, 0]
                y_coords = world[:, 1]
                x_pen = F.relu(-x_coords) + F.relu(x_coords - (grid_x_len - 1))
                y_pen = F.relu(-y_coords) + F.relu(y_coords - (grid_y_len - 1))
                bound_loss = 1.0 * (x_pen.pow(2).sum() + y_pen.pow(2).sum())

                xn = (world[:, 0] / float(Wm1)) * 2.0 - 1.0
                yn = (world[:, 1] / float(Hm1)) * 2.0 - 1.0
                grid_coords = torch.stack([xn, yn], dim=-1).view(1, -1, 1, 2)
                s = F.grid_sample(
                    sdf_torch.to(curr_pos.device),
                    grid_coords,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                in_pen = F.relu(-s.view(-1))
                esc = 2.0 * (in_pen**2).sum() + 8.0 * (in_pen.max() ** 2)

                move_loss = 0.1 * ((curr_pos[i] - init_pos[i]) ** 2).sum()

                total_loss = esc + bound_loss + move_loss + total_loss
                all_inside_counts.append((in_pen > 1e-6).sum().item())

            total_loss.backward()
            optimizer.step()

            if step % 500 == 0:
                logger.log_info(
                    f"[OPT] Step {step} | Total loss {total_loss.item():.3f}"
                )
            # #  debug only
            if (step % visualize_every == 0) or (step == steps - 1):
                try:
                    _, vis_grid = _compute_dynamic_sdf(
                        curr_pos,
                        footprint_offsets_list,
                        grid_bool,
                        grid_x_len,
                        grid_y_len,
                        exclude_idx=None,
                    )
                    # debug only
                    _save_occupancy_snapshot(
                        step=step,
                        curr_pos=curr_pos,
                        footprint_offsets_list=footprint_offsets_list,
                        grid_bool=vis_grid,
                        grid_x_len=grid_x_len,
                        grid_y_len=grid_y_len,
                    )
                    print(f"[DBG] Saved occupancy snapshot for step {step}")
                except Exception as e:
                    print(f"[WARN] Failed to save occupancy snapshot: {e}")

        logger.log_info(
            "Starting post-optimization collision detection (occupancy-based greedy)"
        )
        to_remove_indices = []
        to_remove_uids = []

        final_int = (
            torch.round(curr_pos).detach().cpu().numpy().astype(int)
        )  # [num_units, 2]

        occ_final = grid_bool.copy()

        for i in range(num_units):
            u = opt_units[i]
            uid = u["uid"]

            gx = int(final_int[i, 0])
            gy = int(final_int[i, 1])
            offsets_np = footprint_offsets_list[i].detach().cpu().numpy().astype(int)
            world = offsets_np + np.array([gx, gy], dtype=int)
            x_idx = world[:, 0]
            y_idx = world[:, 1]
            valid_mask = (
                (x_idx >= 0)
                & (x_idx < grid_x_len)
                & (y_idx >= 0)
                & (y_idx < grid_y_len)
            )

            if not valid_mask.any():
                to_remove_indices.append(i)
                to_remove_uids.append(uid)
                logger.log_info(
                    f"Object {uid} out-of-bounds or no valid footprint -> marked for removal"
                )
                continue
            xi = x_idx[valid_mask]
            yi = y_idx[valid_mask]
            if (~occ_final[xi, yi]).any():
                to_remove_indices.append(i)
                to_remove_uids.append(uid)
                logger.log_info(
                    f"[DBG] Object {uid} footprint intersects existing occupied cells -> marked for removal"
                )
                continue

            occ_final[xi, yi] = False

        final_int = torch.round(curr_pos).detach().cpu().numpy()
        placements = {}

        for idx, u in enumerate(opt_units):
            if idx in to_remove_indices:
                continue
            uid = u["uid"]
            gx, gy = float(final_int[idx, 0]), float(final_int[idx, 1])
            gx = float(np.clip(gx, 0.0, grid_x_len - 1.0))
            gy = float(np.clip(gy, 0.0, grid_y_len - 1.0))
            wx = x_min + gx * cell_size
            wy = y_min + gy * cell_size
            wz = z_table + 0.05
            init_pose = uid_to_init_pose.get(uid)
            if init_pose is None:
                local_pose = np.eye(4, dtype=np.float32)
                local_pose[0:3, 3] = np.array([wx, wy, wz], dtype=np.float32)
            else:
                local_pose = init_pose.copy()
                local_pose[0:3, 3] = np.array([wx, wy, wz], dtype=np.float32)
            placements[uid] = {
                "local_pose": local_pose.tolist(),
                "pos": [float(wx), float(wy), float(wz)],
                "is_group": bool(u.get("is_group", False)),
                "bbox": u.get("bbox").tolist(),
                "size": u["size"],
            }

        for obj in objs_data:
            if obj.get("is_group"):
                group_uid = obj["uid"]
                if group_uid in to_remove_uids:
                    continue
                group_place = placements.get(group_uid)
                if group_place is None:
                    continue
                group_center = np.array(group_place["pos"][:2], dtype=np.float32)
                members = obj.get("members", [])
                for m in members:
                    mid = m["uid"]
                    if mid in to_remove_uids:
                        continue
                    offset = m.get("offset_from_group_center", None)
                    if offset is None:
                        new_xy = np.array(m["pos"], dtype=np.float32)
                    else:
                        new_xy = group_center + np.array(offset, dtype=np.float32)

                    wz = z_table + 0.001
                    if "stack_base_height_z" in m:
                        try:
                            wz = (
                                z_table
                                + float(m.get("stack_base_height_z", 0.0))
                                + 0.001
                            )
                        except Exception:
                            wz = z_table + 0.001

                    init_pose = m.get("init_pose", None)
                    if init_pose is None:
                        m_pose = np.eye(4, dtype=np.float32)
                        m_pose[0:3, 3] = np.array(
                            [float(new_xy[0]), float(new_xy[1]), float(wz)],
                            dtype=np.float32,
                        )
                    else:
                        m_pose = init_pose.copy()
                        m_pose[0:3, 3] = np.array(
                            [float(new_xy[0]), float(new_xy[1]), float(wz)],
                            dtype=np.float32,
                        )
                    placements[mid] = {
                        "local_pose": m_pose.tolist(),
                        "pos": [float(new_xy[0]), float(new_xy[1]), float(wz)],
                        "is_group_member_of": group_uid,
                        "bbox": m["bbox"].tolist(),
                        "size": m["size"],
                    }

            if obj.get("locked", False) and not obj.get("group_id"):
                uid = obj["uid"]
                if uid in to_remove_uids:
                    continue
                if uid not in placements:
                    init_pose = obj.get("init_pose", None)
                    if init_pose is not None:
                        placements[uid] = {
                            "local_pose": init_pose.tolist(),
                            "pos": list(init_pose[0:3, 3]),
                            "locked": True,
                            "bbox": obj["bbox"].tolist(),
                            "size": obj["size"],
                        }

        debug = {
            "num_units": num_units,
            "grid_shape": (grid_x_len, grid_y_len),
            "cell_size": cell_size,
            "x_range": (x_min, x_max),
            "y_range": (y_min, y_max),
            "removed_objects": len(to_remove_uids),
            "final_objects": len(placements),
            "removed_uids": to_remove_uids,
        }

        print(
            f"[DBG] Optimization complete, final placements after collision detection: {len(placements)} (removed {len(to_remove_uids)} conflicting objects)"
        )
        return {"status": "ok", "placements": placements, "debug": debug}

    # def __call__(
    #     self,
    #     env: EmbodiedEnv,
    #     env_ids: Union[torch.Tensor, None],
    #     random_percent,
    #     cached_objects,
    # ):
    #     sim = env.sim
    #     cached = deepcopy(cached_objects)

    #     if len(cached) == 0:
    #         logger.log_warning("[] no Cached Object")
    #         return

    #     for uid in env.sim.get_rigid_object_uid_list():
    #         try:
    #             if uid.startswith("dexgen_cached_"):
    #                 sim.get_rigid_object(uid=uid)._entities[0].set_visible(True)
    #                 sim.remove_asset(uid)
    #         except Exception as e:
    #             logger.log_warning(f"[dexgen] remove {uid} failed: {e}")

    #     self._loaded_safe_uids.clear()
    #     total = len(cached)
    #     k = int(round(total * random_percent))
    #     k = max(0, min(total, k))
    #     random.shuffle(cached)
    #     selected = cached[:k]

    #     interact_objs_info, table_scale = self._prepare_initial_state(env, env_ids)
    #     try:
    #         table_info = self._compute_table_avalibale_area(
    #             env, env_ids, interact_objs_info
    #         )

    #         res = self._compute_background_objects_position(
    #             env, env_ids, table_info, selected
    #         )

    #         placements = {}
    #         if isinstance(res, dict):
    #             placements = res.get("placements", {}) or {}

    #         valid_objects = []
    #         for o in selected:
    #             uid = o.get("uid")
    #             if uid and uid in placements:
    #                 bbox = placements[uid].get("bbox")
    #                 if bbox:
    #                     area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
    #                 else:
    #                     area = 0.0
    #                 valid_objects.append((o, area))

    #         valid_objects.sort(key=lambda x: x[1], reverse=True)
    #         sorted_selected = [item[0] for item in valid_objects]

    #         for o in sorted_selected:
    #             uid = o.get("uid")
    #             if not uid:
    #                 continue
    #             if uid not in placements:
    #                 logger.log_info(f"Collision Problem in {uid}, skipped")
    #                 continue

    #             try:
    #                 chosen_pose = None
    #                 if uid in placements:
    #                     try:
    #                         lp = placements[uid].get("local_pose", None)
    #                         if lp is not None:
    #                             chosen_pose = np.array(lp, dtype=np.float32)
    #                     except Exception:
    #                         chosen_pose = None

    #                 if chosen_pose is None:
    #                     chosen_pose = np.array(o.get("init_local_pose"))
    #                 local_pose = torch.tensor(
    #                     chosen_pose, dtype=torch.float32
    #                 ).unsqueeze(0)
    #                 current_object = sim.get_rigid_object(uid=uid)
    #                 current_object.set_local_pose(local_pose)
    #                 com_pose = torch.tensor(
    #                     [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    #                 )
    #                 current_object.set_com_pose(com_pose, env_ids)
    #                 current_object._entities[0].enable_gravity(True)
    #                 current_object.enable_collision(torch.tensor([1]))
    #                 self._loaded_safe_uids.add(uid)
    #                 sim.update(step=500)
    #                 current_object._entities[0].set_visible(True)
    #                 current_object._entities[0].get_physical_body().set_physic_flag(
    #                     dexsim.types.BodyFlag.KINEMATIC, True
    #                 )

    #             except Exception as e:
    #                 logger.log_warning(
    #                     f"[dexgen] Failed for loading {uid}: {e}\n{traceback.format_exc()}"
    #                 )

    #     except:
    #         logger.log_warning(
    #             "dexgen optimize cant work, pls use dexsim==0.3.11 and make sure the RT is available"
    #         )
    #     for obj_info in interact_objs_info:
    #         uid = obj_info["uid"]
    #         original_pose = obj_info["original_pose"]
    #         try:
    #             asset = sim.get_asset(uid)
    #             if asset:
    #                 pose_tensor: torch.Tensor = original_pose.float()
    #                 asset.set_local_pose(pose_tensor)

    #         except Exception as e:
    #             logger.log_warning(
    #                 f"[dexgen] Failed for reload {uid} : {e}\n{traceback.format_exc()}"
    #             )

    #     rx = table_scale[0] - 0.15
    #     robot = env.sim.get_asset("dexforce_w1")
    #     robot.set_local_pose(
    #         torch.tensor([[rx, 0.0, 0.005, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    #     )

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        random_percent,
        cached_objects,
    ):
        sim = env.sim
        cached = deepcopy(cached_objects)

        if len(cached) == 0:
            logger.log_warning("[] no Cached Object")
            return

        for uid in env.sim.get_rigid_object_uid_list():
            try:
                if uid.startswith("dexgen_cached_"):
                    sim.get_rigid_object(uid=uid)._entities[0].set_visible(True)
                    sim.remove_asset(uid)
            except Exception as e:
                logger.log_warning(f"[dexgen] remove {uid} failed: {e}")

        self._loaded_safe_uids.clear()
        total = len(cached)
        k = int(round(total * random_percent))
        k = max(0, min(total, k))
        random.shuffle(cached)
        selected = cached[:k]

        interact_objs_info, table_scale = self._prepare_initial_state(env, env_ids)
        try:
            table_info = self._compute_table_avalibale_area(
                env, env_ids, interact_objs_info
            )

            res = self._compute_background_objects_position(
                env, env_ids, table_info, selected
            )

            placements = {}
            if isinstance(res, dict):
                placements = res.get("placements", {}) or {}

            valid_objects = []
            for o in selected:
                uid = o.get("uid")
                if uid and uid in placements:
                    bbox = placements[uid].get("bbox")
                    if bbox:
                        area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
                    else:
                        area = 0.0
                    valid_objects.append((o, area))

            valid_objects.sort(key=lambda x: x[1], reverse=True)
            sorted_selected = [item[0] for item in valid_objects]

            staged_uids: List[str] = []

            for o in sorted_selected:
                uid = o.get("uid")
                if not uid:
                    continue
                if uid not in placements:
                    logger.log_info(f"Collision Problem in {uid}, skipped")
                    continue

                try:
                    chosen_pose = None
                    try:
                        lp = placements[uid].get("local_pose", None)
                        if lp is not None:
                            chosen_pose = np.array(lp, dtype=np.float32)
                    except Exception:
                        chosen_pose = None

                    if chosen_pose is None:
                        chosen_pose = np.array(o.get("init_local_pose"))

                    local_pose = torch.tensor(
                        chosen_pose, dtype=torch.float32
                    ).unsqueeze(0)
                    current_object = sim.get_rigid_object(uid=uid)
                    current_object.set_local_pose(local_pose)

                    com_pose = torch.tensor(
                        [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
                        dtype=torch.float32,
                    )
                    current_object.set_com_pose(com_pose, env_ids)
                    current_object.enable_collision(torch.tensor([1]))
                    current_object._entities[0].enable_gravity(False)
                    current_object._entities[0].get_physical_body().set_physic_flag(
                        dexsim.types.BodyFlag.KINEMATIC, True
                    )
                    current_object._entities[0].set_visible(False)

                    staged_uids.append(uid)
                    self._loaded_safe_uids.add(uid)

                except Exception as e:
                    logger.log_warning(
                        f"[dexgen] Failed for staging {uid}: {e}\n{traceback.format_exc()}"
                    )

            for uid in staged_uids:
                try:
                    current_object = sim.get_rigid_object(uid=uid)
                    current_object._entities[0].get_physical_body().set_physic_flag(
                        dexsim.types.BodyFlag.KINEMATIC, False
                    )
                    current_object._entities[0].enable_gravity(True)
                    current_object.enable_collision(torch.tensor([1]))
                    current_object._entities[0].set_visible(True)
                except Exception as e:
                    logger.log_warning(f"[dexgen] Failed for releasing {uid}: {e}")

            sim.update(step=500)

            for uid in staged_uids:
                try:
                    current_object = sim.get_rigid_object(uid=uid)
                    current_object._entities[0].get_physical_body().set_physic_flag(
                        dexsim.types.BodyFlag.KINEMATIC, True
                    )
                except Exception:
                    pass

        except:
            logger.log_warning(
                "dexgen optimize cant work, pls use dexsim==0.3.11 and make sure the RT is available"
            )
        for obj_info in interact_objs_info:
            uid = obj_info["uid"]
            original_pose = obj_info["original_pose"]
            try:
                asset = sim.get_asset(uid)
                if asset:
                    pose_tensor: torch.Tensor = original_pose.float()
                    asset.set_local_pose(pose_tensor)

            except Exception as e:
                logger.log_warning(
                    f"[dexgen] Failed for reload {uid} : {e}\n{traceback.format_exc()}"
                )

        rx = table_scale[0] - 0.15
        robot = env.sim.get_asset("dexforce_w1")
        robot.set_local_pose(
            torch.tensor([[rx, 0.0, 0.005, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        )
