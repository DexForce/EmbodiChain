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

from copy import deepcopy
from typing import TYPE_CHECKING, List, Union, Tuple, Dict

from embodichain.lab.sim.objects import (
    Light,
    RigidObject,
    RigidObjectGroup,
    Articulation,
    Robot,
)
from embodichain.lab.sim.cfg import RigidObjectCfg, ArticulationCfg
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.utils.module_utils import find_function_from_modules
from embodichain.utils.string import remove_regex_chars, resolve_matching_names
from embodichain.utils.file import get_all_files_in_directory
from embodichain.utils.math import (
    sample_uniform,
    pose_inv,
    xyz_quat_to_4x4_matrix,
    trans_matrix_to_xyz_quat,
)
from embodichain.utils import logger
from embodichain.data import get_data_path

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


class replace_assets_from_group(Functor):
    """Replace assets in the environment from a specified group of assets.

    The group of assets can be defined in the following ways:
        - A directory containing multiple asset files.
        - A json file listing multiple assets with their properties. (not supported yet)
        - ... (other methods can be added in the future)
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the functor.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        entity_cfg: SceneEntityCfg = cfg.params["entity_cfg"]
        asset = env.sim.get_asset(entity_cfg.uid)
        if asset is None:
            logger.log_error(
                f"Asset with UID '{entity_cfg.uid}' not found in the simulation."
            )

        if (
            isinstance(asset, RigidObject)
            and isinstance(asset.cfg.shape, MeshCfg) is False
        ):
            logger.log_error(
                "Only mesh-based RigidObject assets are supported for replacement."
            )

        self.asset_cfg = asset.cfg
        self.asset_type = type(asset)

        if isinstance(asset, Articulation):
            logger.log_error("Replacing articulation assets is not supported yet.")

        self._asset_group_path: list[str] = []

        # The following block of code only handle rigid object assets.
        # If we want to support articulation assets, the group path format
        # should be changed into list of folder (each folder contains a urdf file
        # and its associated resources)
        folder_path = cfg.params.get("folder_path", None)

        if folder_path is None:
            logger.log_error(
                "folder_path must be specified in the functor configuration."
            )

        if folder_path.endswith("/") is False:
            folder_path, patterns = os.path.split(folder_path)

            # remove regular expression from patterns
            patterns = remove_regex_chars(patterns)
            self._full_path = get_data_path(f"{folder_path}/")
            self._asset_group_path = get_all_files_in_directory(
                self._full_path, patterns=patterns
            )
        else:
            self._full_path = get_data_path(folder_path)
            self._asset_group_path = get_all_files_in_directory(self._full_path)

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        entity_cfg: SceneEntityCfg,
        folder_path: str,
    ) -> None:

        env.sim.remove_asset(entity_cfg.uid)
        asset_path = random.choice(self._asset_group_path)
        self.asset_cfg.shape.fpath = asset_path
        if self.asset_type == RigidObject:
            new_asset = env.sim.add_rigid_object(cfg=self.asset_cfg)
        else:
            logger.log_error("Only RigidObject assets are supported for replacement.")


class prepare_extra_attr(Functor):
    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        """
        Initializes the event manager with the given configuration and environment.

        Args:
            cfg (FunctorCfg): The configuration object for the functor.
            env (EmbodiedEnv): The embodied environment instance.

        Attributes:
            extra_attrs (dict): A dictionary to hold additional attributes.
        """
        super().__init__(cfg, env)

        self.extra_attrs = {}

    def __call__(
        self, env: EmbodiedEnv, env_ids: Union[torch.Tensor, None], attrs: List[Dict]
    ) -> None:
        """
        Processes extra attributes for the given environment.

        This method iterates over a list of attributes, validates them, and updates
        the `extra_attrs` dictionary based on the specified modes and values. It handles
        both static and callable attributes, logging warnings for any issues encountered.

        Args:
            env (EmbodiedEnv): The environment instance to which the attributes are applied.
            env_ids (Union[torch.Tensor, None]): Optional tensor of environment IDs (not used in this method).
            attrs (List[Dict]): A list of dictionaries containing attribute configurations.
                Each dictionary must contain a 'name', and may contain 'entity_cfg', 'entities',
                'mode', 'value', 'func_name', and 'func_kwargs'.

        Returns:
            None: This method does not return a value.
        """
        for attr_idx, attr in enumerate(attrs):
            attr_name = attr.get("name", None)
            if attr_name is None:
                logger.log_warning(
                    f"{attr_idx}-th extra attribute got no name, skipping.."
                )
                continue
            if attr.get("entity_cfg", None) is not None:
                entity_cfgs = [SceneEntityCfg(**attr["entity_cfg"])]
            elif attr.get("entity_uids", None) is not None:
                entity_uids = attr["entity_uids"]
                if isinstance(entity_uids, (str, list)):
                    entity_uids = resolve_uids(env, entity_uids)
                    if entity_uids is None:
                        logger.log_warning(
                            f"Entities string {entity_uids} is not supported, skipping.."
                        )
                        continue
                else:
                    logger.log_warning(
                        f"Entities type {type(entity_uids)} is not supported, skipping.."
                    )
                    continue
                entity_cfgs = [SceneEntityCfg(uid=uid) for uid in entity_uids]
            else:
                logger.log_warning(
                    f"'entity_cfg' or 'entity_uids' must be provieded, skipping.."
                )
                continue

            attr_mode = attr.get("mode", None)
            if attr_mode is None:
                logger.log_info(
                    f"Extra attribute {attr_name} got no mode, setting mode to default 'static'.",
                    color="green",
                )
                attr_mode = "static"

            if attr_mode == "static":
                attr_value = attr.get("value", None)
                if attr_value is None:
                    logger.log_warning(
                        f"Extra attribute {attr_name} got mode 'static' but no value, skipping.."
                    )
                    continue
                for cfg in entity_cfgs:
                    if cfg.uid not in self.extra_attrs:
                        self.extra_attrs[cfg.uid] = {}
                    self.extra_attrs[cfg.uid].update({attr_name: attr_value})

            elif attr_mode == "callable":
                attr_func_name = attr.get("func_name", None)
                if attr_func_name is None:
                    logger.log_info(
                        f"Extra attribute {attr_name} got mode 'callable' but no 'func_name', skipping..",
                        color="green",
                    )
                    continue

                attr_func_kwargs = attr.get("func_kwargs", None)
                if attr_func_name is None:
                    logger.log_info(
                        f"Extra attribute {attr_name} got no func_kwargs, setting func_kwargs to default empty dict..",
                        color="green",
                    )
                    attr_func_kwargs = {}

                is_global_func = True
                ASSET_MODULES = [
                    "embodichain.lab.gym.envs.managers.object",
                    "embodichain.lab.gym.utils.misc",
                ]
                global_func = find_function_from_modules(
                    attr_func_name, modules=ASSET_MODULES, raise_if_not_found=False
                )
                if global_func is None:
                    is_global_func = False
                for cfg in entity_cfgs:
                    if cfg.uid not in self.extra_attrs:
                        self.extra_attrs[cfg.uid] = {}
                    if not is_global_func:
                        asset = env.sim.get_asset(cfg.uid)
                        if callable((attr_func := getattr(asset, attr_func_name))):
                            attr_func_ret = attr_func(**attr_func_kwargs)
                        else:
                            logger.log_warning(
                                f"Extra attribute {attr_name} got no attr_func_name '{attr_func_name}', skipping.."
                            )
                            continue
                    else:
                        attr_func_kwargs.update(
                            {"env": env, "env_ids": env_ids, "entity_cfg": cfg}
                        )
                        attr_func_ret = global_func(**attr_func_kwargs)
                    self.extra_attrs[cfg.uid].update({attr_name: attr_func_ret})


def register_entity_attrs(
    env: EmbodiedEnv,
    env_ids: torch.Tensor,
    entity_cfg: SceneEntityCfg,
    registration: str = "affordance_datas",
    attrs: List[str] = [],
    prefix: bool = True,
):
    """Register the atrributes of an entity to the `env.registration` dict.

    TODO: Currently this method only support 1 env or multi-envs that reset() together,

    as it's behavior is to update a overall dict every time it's called.

    In the future, asynchronously reset mode shall be supported.

    Args:
        env (EmbodiedEnv): The environment the entity is in.
        env_ids (Union[torch.Tensor, None]): The ids of the envs that the entity should be registered.
        entity_cfg (SceneEntityCfg): The config of the entity.
        attrs (List[str]): The list of entity attributes that asked to be registered.
        registration (str, optional): The env's registration string where the attributes should be injected to.
    """
    entity = env.sim.get_asset(entity_cfg.uid)

    if not hasattr(env, registration):
        logger.log_warning(
            f"Environment has no atrtribute {registration} for registration, please check again."
        )
        return
    else:
        registration_dict = getattr(env, registration, None)
        if not isinstance(registration_dict, Dict):
            logger.log_warning(
                f"Got registration env.{registration} with type {type(registration_dict)}, please check again."
            )
            return

    for attr in attrs:
        attr_key = f"{entity_cfg.uid}_{attr}" if prefix else attr
        if (attr_val := getattr(entity, attr_key, None)) is not None:
            registration_dict.update({attr_key: attr_val})
        elif (
            attr_val := getattr(
                env.event_manager.get_functor("prepare_extra_attr"), "extra_attrs", {}
            )
            .get(entity_cfg.uid, {})
            .get(attr)
        ) is not None:
            registration_dict.update({attr_key: attr_val})
        else:
            logger.log_warning(
                f"Attr {attr} for entity {entity_cfg.uid} has neither been found in entity attrbutes nor prepare_extra_attrs functor, skipping.."
            )


def register_entity_pose(
    env: EmbodiedEnv,
    env_ids: torch.Tensor,
    entity_cfg: SceneEntityCfg,
    registration: str = "affordance_datas",
    compute_relative: Union[bool, List, str] = "all_robots",
    compute_pose_object_to_arena: bool = True,
    to_matrix: bool = True,
):
    update_registration_dict = {}
    if not hasattr(env, registration):
        logger.log_warning(
            f"Environment has no atrtribute {registration} for registration, please check again."
        )
        return
    else:
        registration_dict = getattr(env, registration, None)
        if not isinstance(registration_dict, Dict):
            logger.log_warning(
                f"Got registration env.{registration} with type {type(registration_dict)}, please check again."
            )
            return

    entity_pose_name, entity_pose = get_pose(
        env, env_ids, entity_cfg, return_name=True, to_matrix=True
    )
    update_registration_dict.update({entity_pose_name: entity_pose})

    if compute_relative:
        # transform other entity's pose to entity frame
        relative_poses = {}
        if compute_relative == True:
            entity_uids = (
                env.sim.get_articulation_uid_list()
                + env.sim.get_rigid_object_uid_list()
                + env.sim.get_robot_uid_list()
            )
        elif isinstance(compute_relative, (str, list)):
            entity_uids = resolve_uids(env, compute_relative)
        else:
            logger.log_warning(
                f"Compute relative pose option with type {type(compute_relative)} is not supported, using empty list for skipping.."
            )
            entity_uids = []

        for other_entity_uid in entity_uids:
            if other_entity_uid != entity_cfg.uid:
                # TODO: this is only for asset
                other_entity_pose = env.sim.get_asset(other_entity_uid).get_local_pose(
                    to_matrix=True
                )[env_ids, :]
                relative_pose = torch.bmm(pose_inv(entity_pose), other_entity_pose)
                relative_poses.update(
                    {
                        f"{other_entity_uid}_pose_{entity_pose_name.replace('_pose', '')}": relative_pose
                    }
                )

        update_registration_dict.update(relative_poses)

    entity = env.sim.get_asset(entity_cfg.uid)
    if isinstance(entity, RigidObject):
        extra_attr_functor = env.event_manager.get_functor("prepare_extra_attr")
        entity_extra_attrs = getattr(extra_attr_functor, "extra_attrs", {}).get(
            entity_cfg.uid, {}
        )
        for (
            entity_extra_attr_key,
            entity_extra_attr_val,
        ) in entity_extra_attrs.items():
            if entity_extra_attr_key.endswith("_pose_object"):
                entity_extra_attr_val = torch.as_tensor(
                    entity_extra_attr_val, device=env.device
                )
                if entity_extra_attr_val.ndim < 3:
                    logger.log_info(
                        f"Got xyz_quat pose {entity_extra_attr_key}: {entity_extra_attr_val}, transforming it to matrix.",
                        color="green",
                    )
                    entity_extra_attr_val = xyz_quat_to_4x4_matrix(
                        entity_extra_attr_val
                    )
                update_registration_dict.update(
                    {
                        entity_cfg.uid
                        + "_"
                        + (entity_extra_attr_key): entity_extra_attr_val
                    }
                )
                if compute_pose_object_to_arena:
                    pose_arena = torch.bmm(entity_pose, entity_extra_attr_val)
                    update_registration_dict.update(
                        {
                            entity_cfg.uid
                            + "_"
                            + (
                                entity_extra_attr_key.replace("_pose_object", "_pose")
                            ): pose_arena
                        }
                    )
    else:
        logger.log_warning(
            f"Now compute_pose_object_to_arena only support RigidObject type entity, skipping.."
        )

    if not to_matrix:
        for key, val in update_registration_dict.items():
            update_registration_dict[key] = trans_matrix_to_xyz_quat(val)

    registration_dict = getattr(env, registration, None)
    if not isinstance(registration_dict, Dict):
        logger.log_warning(
            f"Got registration env.{registration} with type {type(registration_dict)}, please check again."
        )
        return
    registration_dict.update(update_registration_dict)


def register_info_to_env(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    registry: List[Dict],
    registration: str = "affordance_datas",
    sim_update: bool = True,
):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if sim_update:
        logger.log_info(
            "Calling env.sim.update(100) for after-physics-applied object attributes..",
            color="green",
        )
        env.sim.update(step=100)
    for entity_registry in registry:
        entity_cfg = SceneEntityCfg(**entity_registry["entity_cfg"])
        logger.log_info(f"Registering {entity_cfg.uid}..", color="green")
        if (entity_attrs := entity_registry.get("attrs")) is not None:
            prefix = entity_registry.get("prefix", True)
            register_entity_attrs(
                env, env_ids, entity_cfg, registration, entity_attrs, prefix
            )
        if (
            pose_register_params := entity_registry.get("pose_register_params")
        ) is not None:
            register_entity_pose(
                env, env_ids, entity_cfg, registration, **pose_register_params
            )


"""Helper Function"""


def resolve_uids(env: EmbodiedEnv, entity_uids: Union[List[str], str]) -> List[str]:
    if isinstance(entity_uids, str):
        if entity_uids == "all_objects":
            entity_uids = (
                env.sim.get_rigid_object_uid_list()
                + env.sim.get_articulation_uid_list()
            )
        elif entity_uids == "all_robots":
            entity_uids = env.sim.get_robot_uid_list()
        elif entity_uids == "all_sensors":
            entity_uids = env.sim.get_sensor_uid_list()
        else:
            # logger.log_warning(f"Entity uids {entity_uids} not supported in ['all_objects', 'all_robots', 'all_sensors'], wrapping it as a list..")
            entity_uids = [entity_uids]
    elif isinstance(entity_uids, (list, set, tuple)):
        entity_uids = list(entity_uids)
    else:
        logger.log_error(
            f"Entity uids {entity_uids} with type {type(entity_uids)} not supported in [List[str], str], please check again."
        )
    return entity_uids


def resolve_dict(env: EmbodiedEnv, entity_dict: Dict):
    for entity_key in list(entity_dict.keys()):
        entity_val = entity_dict.pop(entity_key)
        entity_uids = resolve_uids(env, entity_key)
        for entity_uid in entity_uids:
            entity_dict.update({entity_uid: deepcopy(entity_val)})
    return entity_dict


EntityWithPose = Union[RigidObject, Robot]


def get_pose(
    env: EmbodiedEnv,
    env_ids: torch.Tensor,
    entity_cfg: SceneEntityCfg,
    return_name: bool = True,
    to_matrix: bool = True,
):
    entity = env.sim.get_asset(entity_cfg.uid)

    if isinstance(entity, RigidObject):
        entity_pose = entity.get_local_pose(to_matrix=to_matrix)[env_ids, :]
        entity_pose_register_name = entity_cfg.uid + "_pose"
    elif isinstance(entity, Robot):
        _, control_parts = resolve_matching_names(
            entity_cfg.control_parts, list(entity.control_parts.keys())
        )
        if len(control_parts) != 1:
            logger.log_warning(
                "Only 1 control part can be assigned for computing the robot pose, please check again. Skipping"
            )
            return None
        entity_cfg.control_parts = control_parts
        control_part = control_parts[0]
        control_part_qpos = entity.get_qpos()[
            env_ids, entity.get_joint_ids(control_part)
        ]
        entity_pose = entity.compute_fk(
            control_part_qpos, name=control_part, to_matrix=to_matrix
        )  # NOTE: now compute_fk returns arena pose
        entity_pose_register_name = control_part + "_pose"
    else:
        logger.log_warning(
            f"Entity with tyope {type(entity)} is not supported, please check again."
        )
        return None

    if return_name:
        return entity_pose_register_name, entity_pose
    else:
        return entity_pose


def drop_rigid_object_group_sequentially(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    drop_position: List[float] = [0.0, 0.0, 1.0],
    position_range: Tuple[List[float], List[float]] = (
        [-0.1, -0.1, 0.0],
        [0.1, 0.1, 0.0],
    ),
    physics_step: int = 2,
) -> None:
    """Drop rigid object group from a specified height sequentially in the environment.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (Union[torch.Tensor, None]): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        drop_position (List[float]): The base position from which to drop the objects. Default is [0.0, 0.0, 1.0].
        position_range (Tuple[List[float], List[float]]): The range for randomizing the drop position around the base position.
        physics_step (int): The number of physics steps to simulate after dropping the objects. Default is 2.
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
        )
        drop_pose_i = drop_pose.unsqueeze(1)
        drop_pose_i[:, 0, :3] = drop_pos + random_offset

        obj_group.set_local_pose(pose=drop_pose_i, env_ids=env_ids, obj_ids=[i])

        env.sim.update(step=physics_step)
