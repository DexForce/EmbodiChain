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
import os
import random
from collections.abc import Sequence
from copy import deepcopy
from numbers import Real
from typing import TYPE_CHECKING, Dict, List, Literal, Tuple

import numpy as np
import torch

from embodichain.lab.sim.objects import (
    Light,
    RigidObject,
    RigidObjectGroup,
    Articulation,
    Robot,
)
from embodichain.lab.sim.cfg import RigidObjectCfg, ArticulationCfg, RigidConstraintCfg
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


__all__ = [
    "replace_assets_from_group",
    "prepare_extra_attr",
    "register_entity_attrs",
    "register_entity_pose",
    "register_info_to_env",
    "resolve_uids",
    "resolve_dict",
    "get_pose",
    "drop_rigid_object_group_sequentially",
    "wait_for_dynamic_objects_to_settle",
    "set_detached_uids_for_env_reset",
    "create_rigid_constraint",
    "remove_rigid_constraint",
]

_DynamicEntity = RigidObject | RigidObjectGroup | Articulation
_SettleEntity = tuple[str, SceneEntityCfg, _DynamicEntity]
_SpeedSample = tuple[str, torch.Tensor, torch.Tensor]


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
        env_ids: torch.Tensor | None,
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
        self, env: EmbodiedEnv, env_ids: torch.Tensor | None, attrs: List[Dict]
    ) -> None:
        """
        Processes extra attributes for the given environment.

        This method iterates over a list of attributes, validates them, and updates
        the `extra_attrs` dictionary based on the specified modes and values. It handles
        both static and callable attributes, logging warnings for any issues encountered.

        Args:
            env (EmbodiedEnv): The environment instance to which the attributes are applied.
            env_ids (torch.Tensor | None): Optional tensor of environment IDs (not used in this method).
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
        env_ids (torch.Tensor | None): The ids of the envs that the entity should be registered.
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
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    registration: str = "affordance_datas",
    compute_relative: bool | List | str = "all_robots",
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
    env_ids: torch.Tensor | None,
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


def resolve_uids(env: EmbodiedEnv, entity_uids: list[str] | str) -> list[str]:
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
    env_ids: torch.Tensor | None,
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
        env_ids (torch.Tensor | None): The environment IDs to apply the event.
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
            device=env.device,
        )
        drop_pose_i = drop_pose.unsqueeze(1)
        drop_pose_i[:, 0, :3] = drop_pos + random_offset

        obj_group.set_local_pose(pose=drop_pose_i, env_ids=env_ids, obj_ids=[i])

        env.sim.update(step=physics_step)


def _validate_settle_parameters(
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    min_steps: int,
    max_steps: int,
    check_interval_steps: int,
    required_stable_checks: int,
    timeout_behavior: str,
    allow_partial_envs: bool,
) -> None:
    """Validate dynamic-object settle parameters."""
    for name, value in (
        ("min_steps", min_steps),
        ("max_steps", max_steps),
        ("check_interval_steps", check_interval_steps),
        ("required_stable_checks", required_stable_checks),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")

    if min_steps < 0:
        raise ValueError("min_steps must be non-negative.")
    if max_steps < min_steps:
        raise ValueError("max_steps must be greater than or equal to min_steps.")
    if check_interval_steps < 1:
        raise ValueError("check_interval_steps must be at least 1.")
    if required_stable_checks < 1:
        raise ValueError("required_stable_checks must be at least 1.")

    for name, value in (
        ("linear_velocity_threshold", linear_velocity_threshold),
        ("angular_velocity_threshold", angular_velocity_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number.")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")

    available_checks = (
        1 + (max_steps - min_steps + check_interval_steps - 1) // check_interval_steps
    )
    if required_stable_checks > available_checks:
        raise ValueError(
            "required_stable_checks cannot be reached within the configured "
            f"step budget; at most {available_checks} checks are possible."
        )

    if timeout_behavior not in ("warn", "raise"):
        raise ValueError("timeout_behavior must be either 'warn' or 'raise'.")
    if not isinstance(allow_partial_envs, bool):
        raise TypeError("allow_partial_envs must be a boolean.")


def _normalize_settle_env_ids(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | Sequence[int] | slice | None,
) -> torch.Tensor:
    """Normalize environment IDs into a unique one-dimensional tensor."""
    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if env_ids is None:
        return all_env_ids
    if isinstance(env_ids, slice):
        return all_env_ids[env_ids]

    raw_env_ids = torch.as_tensor(env_ids, device=env.device)
    if raw_env_ids.dtype == torch.bool:
        raise TypeError("env_ids must contain integer indices, not booleans.")
    if raw_env_ids.ndim == 0:
        raw_env_ids = raw_env_ids.unsqueeze(0)
    elif raw_env_ids.ndim != 1:
        raise ValueError("env_ids must be a one-dimensional sequence of indices.")
    if raw_env_ids.is_floating_point() and not bool(
        torch.equal(raw_env_ids, raw_env_ids.round())
    ):
        raise ValueError("env_ids must contain integer-valued indices.")

    normalized = torch.unique(raw_env_ids.to(dtype=torch.long), sorted=True)
    if normalized.numel() > 0 and bool(
        ((normalized < 0) | (normalized >= env.num_envs)).any().item()
    ):
        raise IndexError(f"env_ids must be within [0, {env.num_envs - 1}].")
    return normalized


def _get_dynamic_entity_catalog(
    env: EmbodiedEnv,
) -> dict[str, tuple[str, _DynamicEntity]]:
    """Collect settle-capable non-robot scene entities by UID."""
    catalog: dict[str, tuple[str, _DynamicEntity]] = {}

    for uid in env.sim.get_rigid_object_uid_list():
        entity = env.sim.get_rigid_object(uid)
        if entity is not None:
            catalog[uid] = ("rigid_object", entity)
    for uid in env.sim.get_rigid_object_group_uid_list():
        entity = env.sim.get_rigid_object_group(uid)
        if entity is not None:
            catalog[uid] = ("rigid_object_group", entity)
    for uid in env.sim.get_articulation_uid_list():
        entity = env.sim.get_articulation(uid)
        if entity is not None:
            catalog[uid] = ("articulation", entity)

    return catalog


def _is_dynamic_entity(kind: str, entity: _DynamicEntity) -> bool:
    """Return whether an entity participates in dynamic physics."""
    if kind == "articulation":
        return getattr(entity.cfg, "body_type", None) == "dynamic"
    return not entity.is_non_dynamic


def _resolve_settle_entities(
    env: EmbodiedEnv,
    entity_cfgs: Sequence[SceneEntityCfg] | None,
) -> list[_SettleEntity]:
    """Resolve explicit or automatically discovered dynamic entities."""
    catalog = _get_dynamic_entity_catalog(env)
    explicit = entity_cfgs is not None
    if entity_cfgs is None:
        configs = [SceneEntityCfg(uid=uid) for uid in catalog]
    else:
        configs = list(entity_cfgs)
        if not configs:
            raise ValueError("entity_cfgs must not be empty when provided.")

    resolved: list[_SettleEntity] = []
    seen_uids: set[str] = set()
    robot_uids = set(env.sim.get_robot_uid_list())
    for entity_cfg in configs:
        if not isinstance(entity_cfg, SceneEntityCfg):
            raise TypeError(
                "entity_cfgs must contain only SceneEntityCfg instances, got "
                f"{type(entity_cfg).__name__}."
            )
        if entity_cfg.uid in seen_uids:
            continue
        seen_uids.add(entity_cfg.uid)

        catalog_entry = catalog.get(entity_cfg.uid)
        if catalog_entry is None:
            if entity_cfg.uid in robot_uids:
                raise ValueError(
                    f"Robot '{entity_cfg.uid}' cannot be used as a settle target."
                )
            raise ValueError(
                f"Settle target '{entity_cfg.uid}' is not a rigid object, rigid "
                "object group, or articulation."
            )

        kind, entity = catalog_entry
        if not _is_dynamic_entity(kind, entity):
            if explicit:
                raise ValueError(
                    f"Settle target '{entity_cfg.uid}' is static or kinematic."
                )
            continue
        resolved.append((kind, entity_cfg, entity))

    return resolved


def _measure_settle_speeds(
    entities: Sequence[_SettleEntity],
    env_ids: torch.Tensor,
) -> list[_SpeedSample]:
    """Measure per-body linear and angular speeds for selected environments."""
    samples: list[_SpeedSample] = []
    for kind, entity_cfg, entity in entities:
        if kind == "articulation":
            velocity = entity.body_data.body_link_vel[env_ids]
            velocity = velocity[:, entity_cfg.body_ids, :]
            linear_velocity = velocity[..., :3]
            angular_velocity = velocity[..., 3:]
        else:
            body_data = entity.body_data
            if body_data is None:
                raise RuntimeError(
                    f"Dynamic settle target '{entity_cfg.uid}' has no body data."
                )
            linear_velocity = body_data.lin_vel[env_ids]
            angular_velocity = body_data.ang_vel[env_ids]

        if linear_velocity.numel() == 0 or angular_velocity.numel() == 0:
            raise ValueError(
                f"Settle target '{entity_cfg.uid}' selected no physical bodies."
            )
        linear_speed = torch.linalg.vector_norm(linear_velocity, dim=-1).reshape(
            env_ids.numel(), -1
        )
        angular_speed = torch.linalg.vector_norm(angular_velocity, dim=-1).reshape(
            env_ids.numel(), -1
        )
        samples.append((entity_cfg.uid, linear_speed, angular_speed))
    return samples


def _settle_samples_are_stable(
    samples: Sequence[_SpeedSample],
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
) -> bool:
    """Return whether every measured body is finite and below both thresholds."""
    stable = []
    for _, linear_speed, angular_speed in samples:
        stable.append(
            torch.isfinite(linear_speed)
            & torch.isfinite(angular_speed)
            & (linear_speed <= linear_velocity_threshold)
            & (angular_speed <= angular_velocity_threshold)
        )
    return bool(torch.cat([value.reshape(-1) for value in stable]).all().item())


def _format_settle_timeout(
    samples: Sequence[_SpeedSample],
    env_ids: torch.Tensor,
    linear_velocity_threshold: float,
    angular_velocity_threshold: float,
    max_steps: int,
    stable_checks: int,
    required_stable_checks: int,
) -> str:
    """Build a timeout message with per-entity environment diagnostics."""
    unsettled: list[str] = []
    all_linear_speeds: list[torch.Tensor] = []
    all_angular_speeds: list[torch.Tensor] = []
    for uid, linear_speed, angular_speed in samples:
        stable = (
            torch.isfinite(linear_speed)
            & torch.isfinite(angular_speed)
            & (linear_speed <= linear_velocity_threshold)
            & (angular_speed <= angular_velocity_threshold)
        )
        unsettled_mask = ~stable.all(dim=1)
        if bool(unsettled_mask.any().item()):
            unsettled_env_ids = env_ids[unsettled_mask].detach().cpu().tolist()
            unsettled.append(f"{uid}(env_ids={unsettled_env_ids})")
        all_linear_speeds.append(linear_speed.reshape(-1))
        all_angular_speeds.append(angular_speed.reshape(-1))

    linear_speeds = torch.cat(all_linear_speeds)
    angular_speeds = torch.cat(all_angular_speeds)
    infinity = torch.tensor(float("inf"), device=linear_speeds.device)
    max_linear_speed = torch.where(
        torch.isfinite(linear_speeds), linear_speeds, infinity
    ).max()
    infinity = infinity.to(device=angular_speeds.device)
    max_angular_speed = torch.where(
        torch.isfinite(angular_speeds), angular_speeds, infinity
    ).max()
    unsettled_summary = ", ".join(unsettled) or "none at the final check"
    return (
        f"Dynamic objects did not settle within {max_steps} physics steps. "
        f"Stable checks: {stable_checks}/{required_stable_checks}; unsettled: "
        f"{unsettled_summary}; maximum linear speed: "
        f"{max_linear_speed.item():.6g} m/s; maximum angular speed: "
        f"{max_angular_speed.item():.6g} rad/s."
    )


def wait_for_dynamic_objects_to_settle(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | Sequence[int] | slice | None,
    entity_cfgs: Sequence[SceneEntityCfg] | None = None,
    linear_velocity_threshold: float = 0.03,
    angular_velocity_threshold: float = 0.20,
    min_steps: int = 10,
    max_steps: int = 240,
    check_interval_steps: int = 2,
    required_stable_checks: int = 3,
    timeout_behavior: Literal["warn", "raise"] = "warn",
    allow_partial_envs: bool = False,
) -> None:
    """Advance physics until selected dynamic objects remain stationary.

    The functor waits at least ``min_steps`` and then polls every
    ``check_interval_steps``. Every selected body in every selected environment
    must remain below both velocity thresholds for
    ``required_stable_checks`` consecutive polls. It never clears dynamics and
    never advances beyond ``max_steps``.

    When ``entity_cfgs`` is ``None``, all dynamic rigid objects, rigid object
    groups, and non-robot articulations are selected automatically. Static and
    kinematic entities are ignored during automatic discovery.

    .. attention::
        :meth:`SimulationManager.update` advances the entire vectorized physics
        world. Partial ``env_ids`` are therefore rejected by default. Set
        ``allow_partial_envs=True`` only when advancing non-target environments
        is acceptable.

    Args:
        env: The environment instance.
        env_ids: Target environment IDs. ``None`` or ``slice(None)`` selects all
            environments.
        entity_cfgs: Explicit settle targets. ``None`` discovers all supported
            dynamic non-robot entities.
        linear_velocity_threshold: Maximum stable linear speed in meters per
            second.
        angular_velocity_threshold: Maximum stable angular speed in radians per
            second.
        min_steps: Physics steps to run before the first stability check.
        max_steps: Maximum total number of physics steps to run.
        check_interval_steps: Physics steps between stability checks.
        required_stable_checks: Consecutive stable checks required before return.
        timeout_behavior: ``"warn"`` to log and continue or ``"raise"`` to
            raise :class:`TimeoutError` when ``max_steps`` is reached.
        allow_partial_envs: Whether to permit a partial environment selection
            despite whole-world physics advancement.

    Raises:
        IndexError: If an environment ID is outside the valid range.
        RuntimeError: If a dynamic target has no readable body data.
        TimeoutError: If objects do not settle and ``timeout_behavior`` is
            ``"raise"``.
        TypeError: If a parameter or entity configuration has the wrong type.
        ValueError: If parameters, targets, or environment selection are invalid.
    """
    _validate_settle_parameters(
        linear_velocity_threshold=linear_velocity_threshold,
        angular_velocity_threshold=angular_velocity_threshold,
        min_steps=min_steps,
        max_steps=max_steps,
        check_interval_steps=check_interval_steps,
        required_stable_checks=required_stable_checks,
        timeout_behavior=timeout_behavior,
        allow_partial_envs=allow_partial_envs,
    )
    target_env_ids = _normalize_settle_env_ids(env, env_ids)
    if target_env_ids.numel() == 0:
        return

    all_env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    if not allow_partial_envs and not torch.equal(target_env_ids, all_env_ids):
        raise ValueError(
            "Partial env_ids would still advance every environment in the physics "
            "world. Pass allow_partial_envs=True to accept this side effect."
        )

    entities = _resolve_settle_entities(env, entity_cfgs)
    if not entities:
        logger.log_warning("No dynamic objects were found to settle.")
        return

    step_count = 0
    if min_steps > 0:
        env.sim.update(step=min_steps)
        step_count = min_steps

    stable_checks = 0
    samples: list[_SpeedSample]
    while True:
        samples = _measure_settle_speeds(entities, target_env_ids)
        if _settle_samples_are_stable(
            samples,
            linear_velocity_threshold=linear_velocity_threshold,
            angular_velocity_threshold=angular_velocity_threshold,
        ):
            stable_checks += 1
            if stable_checks >= required_stable_checks:
                return
        else:
            stable_checks = 0

        if step_count >= max_steps:
            break
        update_steps = min(check_interval_steps, max_steps - step_count)
        env.sim.update(step=update_steps)
        step_count += update_steps

    timeout_message = _format_settle_timeout(
        samples=samples,
        env_ids=target_env_ids,
        linear_velocity_threshold=linear_velocity_threshold,
        angular_velocity_threshold=angular_velocity_threshold,
        max_steps=max_steps,
        stable_checks=stable_checks,
        required_stable_checks=required_stable_checks,
    )
    if timeout_behavior == "raise":
        raise TimeoutError(timeout_message)
    logger.log_warning(timeout_message)


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


def create_rigid_constraint(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    obj_a_cfg: SceneEntityCfg,
    obj_b_cfg: SceneEntityCfg,
    name: str,
    local_frame_a: np.ndarray | None = None,
    local_frame_b: np.ndarray | None = None,
) -> None:
    """Attach two rigid objects via a fixed constraint for the given env_ids.

    Registered under a custom event mode (e.g. ``"attach"``); the task triggers it
    with ``env.event_manager.apply(mode="attach", env_ids=...)``. Delegates to
    :meth:`SimulationManager.create_rigid_constraint`.

    Args:
        env: The environment instance.
        env_ids: Target environment indices. None -> all envs.
        obj_a_cfg: SceneEntityCfg pointing at the first RigidObject.
        obj_b_cfg: SceneEntityCfg pointing at the second RigidObject.
        name: Base constraint name; per-arena names derived by the sim layer.
        local_frame_a: Local joint frame on object A. None -> identity (object
            A's origin). Accepts (4,4) or (N,4,4).
        local_frame_b: Local joint frame on object B. None -> computed per env as
            ``inv(pose_B) @ pose_A`` so the constraint welds the objects at their
            current relative pose. Accepts (4,4) or (N,4,4).

    Raises:
        RuntimeError: If either entity is not a RigidObject.
    """
    obj_a = env.sim.get_asset(obj_a_cfg.uid)
    obj_b = env.sim.get_asset(obj_b_cfg.uid)
    if not isinstance(obj_a, RigidObject) or not isinstance(obj_b, RigidObject):
        logger.log_error(
            f"Constraint '{name}' requires two RigidObjects, but got "
            f"{type(obj_a).__name__} and {type(obj_b).__name__}."
        )
    env.sim.create_rigid_constraint(
        cfg=RigidConstraintCfg(
            name=name,
            rigid_object_a_uid=obj_a_cfg.uid,
            rigid_object_b_uid=obj_b_cfg.uid,
            local_frame_a=local_frame_a,
            local_frame_b=local_frame_b,
        ),
        env_ids=env_ids,
    )


def remove_rigid_constraint(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    name: str,
) -> None:
    """Remove the named constraint for the given env_ids.

    Delegates to :meth:`SimulationManager.remove_rigid_constraint`. Idempotent:
    warns (via the sim layer) if the constraint is not found.

    Args:
        env: The environment instance.
        env_ids: Target environment indices. None -> all envs.
        name: Base constraint name to remove.
    """
    env.sim.remove_rigid_constraint(name, env_ids=env_ids)
