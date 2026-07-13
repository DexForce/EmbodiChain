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
import os
import random
import copy
import numpy as np
from pathlib import Path

from typing import TYPE_CHECKING, Literal, Union, Dict, Sequence, Mapping

from embodichain.lab.sim.objects import (
    Light,
    RigidObject,
    Articulation,
    RigidObjectGroup,
)
from embodichain.lab.sim.sensors import Camera, StereoCamera
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg
from embodichain.lab.sim import (
    VisualMaterial,
    VisualMaterialInst,
    VisualMaterialCfg,
)
from embodichain.utils.string import resolve_matching_names
from embodichain.utils.math import (
    sample_uniform,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
)
from embodichain.utils import logger
from embodichain.data import get_data_path

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


__all__ = [
    "randomize_camera_extrinsics",
    "randomize_light",
    "randomize_emission_light",
    "randomize_camera_intrinsics",
    "set_rigid_object_visual_material",
    "set_rigid_object_group_visual_material",
    "randomize_visual_material",
    "randomize_indirect_lighting",
]


def _get_texture_refs(sim, key: str, textures: list[torch.Tensor]) -> list[object]:
    """Return simulation-cached GPU texture references for source images."""
    cached = sim.get_texture_ref_cache(key)
    if cached is not None:
        return cached
    import dexsim

    dex_env = dexsim.default_world().get_env()
    refs = [dex_env.create_color_texture(image.cpu().numpy(), has_alpha=True) for image in textures]
    sim.set_texture_ref_cache(key, refs)
    return refs


def _select_texture_indices(
    mode: str,
    env_ids: Sequence[int],
    count: int,
    texture_indices: Mapping[int, int] | None = None,
) -> list[int]:
    if mode == "random":
        return torch.randint(0, count, (len(env_ids),)).tolist() if count else []
    if mode == "without_replacement":
        if len(env_ids) > count:
            raise ValueError(
                "without_replacement requires at least one texture per environment"
            )
        return torch.randperm(count)[: len(env_ids)].tolist()
    if mode == "fixed":
        if texture_indices is None or any(i not in texture_indices for i in env_ids):
            raise ValueError(
                "fixed texture_sampling requires texture_indices for every environment"
            )
        result = [int(texture_indices[i]) for i in env_ids]
        if any(i < 0 or i >= count for i in result):
            raise ValueError("texture index out of range")
        return result
    if mode == "cycle":
        return [i % count for i in env_ids] if count else []
    raise ValueError(f"Unsupported texture_sampling mode: {mode}")


def _normalize_env_ids(
    env_ids: torch.Tensor | Sequence[int] | slice | None, num_envs: int
) -> list[int]:
    """Normalize environment selectors to validated integer IDs."""
    if num_envs < 0:
        raise ValueError("num_envs must be non-negative")
    if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
        ids = list(range(num_envs))
    elif isinstance(env_ids, slice):
        ids = list(range(num_envs))[env_ids]
    elif isinstance(env_ids, torch.Tensor):
        ids = env_ids.detach().cpu().reshape(-1).tolist()
    else:
        try:
            ids = list(env_ids)
        except TypeError:
            ids = [env_ids]
    normalized: list[int] = []
    for env_id in ids:
        if isinstance(env_id, bool) or int(env_id) != env_id:
            raise ValueError(f"Invalid environment ID: {env_id!r}")
        env_id = int(env_id)
        if env_id < 0 or env_id >= num_envs:
            raise ValueError(f"Environment ID {env_id} is outside [0, {num_envs})")
        normalized.append(env_id)
    return normalized


def _select_texture_indices(
    mode: str,
    target_ids: list[int],
    num_textures: int,
    texture_indices: dict[int, int] | None,
) -> list[int]:
    """Select one texture index for each target environment."""
    if num_textures <= 0:
        raise ValueError("At least one texture source is required")
    count = len(target_ids)
    if mode == "random":
        return torch.randint(num_textures, (count,)).tolist()
    if mode == "without_replacement":
        if count > num_textures:
            raise ValueError(
                "without_replacement requires at least one texture per target"
            )
        return torch.randperm(num_textures)[:count].tolist()
    if mode == "cycle":
        return [index % num_textures for index in range(count)]
    if mode == "fixed":
        if texture_indices is None:
            raise ValueError("fixed texture selection requires texture_indices")
        selected: list[int] = []
        for env_id in target_ids:
            if env_id not in texture_indices:
                raise ValueError(
                    f"Missing fixed texture index for environment {env_id}"
                )
            index = texture_indices[env_id]
            if (
                isinstance(index, bool)
                or int(index) != index
                or not 0 <= int(index) < num_textures
            ):
                raise ValueError(
                    f"Invalid texture index for environment {env_id}: {index!r}"
                )
            selected.append(int(index))
        return selected
    raise ValueError(f"Unknown texture selection mode: {mode!r}")


def set_rigid_object_visual_material(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    mat_cfg: Union[VisualMaterialCfg, Dict],
) -> None:
    """Set a rigid object's visual material (deterministic, non-random).

    This helper exists to support configs that want fixed colors/materials during reset.

    Args:
        env: Environment instance.
        env_ids: Target env ids. If None, applies to all envs.
        entity_cfg: Scene entity config (must point to a rigid object).
        mat_cfg: Visual material configuration. Can be a VisualMaterialCfg object or a dict.
            If a dict is provided, it will be converted to VisualMaterialCfg using from_dict().
            If uid is not specified in mat_cfg, it will default to "{entity_uid}_mat".
    """
    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if isinstance(mat_cfg, dict):
        mat_cfg = VisualMaterialCfg.from_dict(mat_cfg)

    mat_cfg = copy.deepcopy(mat_cfg)

    if not mat_cfg.uid or mat_cfg.uid == "default_mat":
        mat_cfg.uid = f"{entity_cfg.uid}_mat"

    mat = env.sim.create_visual_material(mat_cfg)
    obj: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)
    obj.set_visual_material(mat, env_ids=env_ids)


def set_rigid_object_group_visual_material(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | None,
    entity_cfg: SceneEntityCfg,
    mat_cfg: VisualMaterialCfg | Dict,
) -> None:
    """Set a rigid object group's visual material (deterministic, non-random).

    This helper exists to support configs that want fixed colors/materials during reset.

    Args:
        env: Environment instance.
        env_ids: Target env ids. If None, applies to all envs.
        entity_cfg: Scene entity config (must point to a rigid object).
        mat_cfg: Visual material configuration. Can be a VisualMaterialCfg object or a dict.
            If a dict is provided, it will be converted to VisualMaterialCfg using from_dict().
            If uid is not specified in mat_cfg, it will default to "{entity_uid}_mat".
    """
    if entity_cfg.uid not in env.sim.get_rigid_object_group_uid_list():
        return

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if isinstance(mat_cfg, dict):
        mat_cfg = VisualMaterialCfg.from_dict(mat_cfg)

    mat_cfg = copy.deepcopy(mat_cfg)

    if not mat_cfg.uid or mat_cfg.uid == "default_mat":
        mat_cfg.uid = f"{entity_cfg.uid}_mat"

    mat = env.sim.create_visual_material(mat_cfg)
    obj: RigidObjectGroup = env.sim.get_rigid_object_group(entity_cfg.uid)
    obj.set_visual_material(mat, env_ids=env_ids)


def randomize_camera_extrinsics(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    pos_range: tuple[list[float], list[float]] | None = None,
    euler_range: tuple[list[float], list[float]] | None = None,
    eye_range: tuple[list[float], list[float]] | None = None,
    target_range: tuple[list[float], list[float]] | None = None,
    up_range: tuple[list[float], list[float]] | None = None,
) -> None:
    """
    Randomize camera extrinsic properties (position and orientation).

    Behavior:
    - If extrinsics config has a parent field (attach mode), pos_range/euler_range are used to perturb the initial pose (pos, quat),
        and set_local_pose is called to attach the camera to the parent node. In this case, pose is related to parent.
    - If extrinsics config uses eye/target/up (no parent), eye_range/target_range/up_range are used to perturb the initial eye, target, up vectors,
        and look_at is called to set the camera orientation.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        pos_range: Position perturbation range (attach mode).
        euler_range: Euler angle perturbation range (attach mode).
        eye_range: Eye position perturbation range (look_at mode).
        target_range: Target position perturbation range (look_at mode).
        up_range: Up vector perturbation range (look_at mode).
    """
    camera: Union[Camera, StereoCamera] = env.sim.get_sensor(entity_cfg.uid)
    num_instance = len(env_ids)

    extrinsics = camera.cfg.extrinsics

    if extrinsics.parent is not None:
        # If extrinsics has a parent field, use pos/euler perturbation and attach camera to parent node
        init_pos = getattr(extrinsics, "pos", [0.0, 0.0, 0.0])
        init_quat = getattr(extrinsics, "quat", [0.0, 0.0, 0.0, 1.0])
        new_pose = torch.tensor(
            [init_pos + init_quat], dtype=torch.float32, device=env.device
        ).repeat(num_instance, 1)
        if pos_range:
            random_value = sample_uniform(
                lower=torch.tensor(pos_range[0]),
                upper=torch.tensor(pos_range[1]),
                size=(num_instance, 3),
            )
            new_pose[:, :3] += random_value
        if euler_range:
            # 1. quat -> euler
            init_quat_np = (
                torch.tensor(init_quat, dtype=torch.float32, device=env.device)
                .unsqueeze_(0)
                .repeat(num_instance, 1)
            )
            init_euler = torch.stack(euler_xyz_from_quat(init_quat_np), dim=1)
            # 2. Sample perturbation for euler angles
            random_value = sample_uniform(
                lower=torch.tensor(euler_range[0]),
                upper=torch.tensor(euler_range[1]),
                size=(num_instance, 3),
            )
            # 3. Add perturbation to each environment and convert back to quaternion
            roll, pitch, yaw = (init_euler + random_value).unbind(dim=1)
            new_quat = quat_from_euler_xyz(roll, pitch, yaw)
            new_pose[:, 3:7] = new_quat

        camera.set_local_pose(new_pose, env_ids=env_ids)

    elif extrinsics.eye is not None:
        # If extrinsics uses eye/target/up, use perturbation for look_at mode
        init_eye = (
            torch.tensor(extrinsics.eye, dtype=torch.float32, device=env.device)
            .unsqueeze(0)
            .repeat(num_instance, 1)
        )
        init_target = (
            torch.tensor(extrinsics.target, dtype=torch.float32, device=env.device)
            .unsqueeze(0)
            .repeat(num_instance, 1)
        )
        init_up = (
            torch.tensor(extrinsics.up, dtype=torch.float32, device=env.device)
            .unsqueeze(0)
            .repeat(num_instance, 1)
        )

        if eye_range:
            eye_delta = sample_uniform(
                lower=torch.tensor(eye_range[0]),
                upper=torch.tensor(eye_range[1]),
                size=(num_instance, 3),
            )
            new_eye = init_eye + eye_delta
        else:
            new_eye = init_eye

        if target_range:
            target_delta = sample_uniform(
                lower=torch.tensor(target_range[0]),
                upper=torch.tensor(target_range[1]),
                size=(num_instance, 3),
            )
            new_target = init_target + target_delta
        else:
            new_target = init_target

        if up_range:
            up_delta = sample_uniform(
                lower=torch.tensor(up_range[0]),
                upper=torch.tensor(up_range[1]),
                size=(num_instance, 3),
            )
            new_up = init_up + up_delta
        else:
            new_up = init_up

        camera.look_at(new_eye, new_target, new_up, env_ids=env_ids)

    else:
        logger.log_error("Unsupported extrinsics format for camera randomization.")


def randomize_light(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    position_range: tuple[list[float], list[float]] | None = None,
    color_range: tuple[list[float], list[float]] | None = None,
    intensity_range: tuple[float, float] | None = None,
    direction_range: tuple[list[float], list[float]] | None = None,
) -> None:
    """Randomize light properties by adding, scaling, or setting random values.

    This function allows randomizing light properties in the scene. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the
    operation.

    The distribution parameters are lists of two elements each, representing the lower and upper bounds of the
    distribution for the x, y, and z components of the light properties. The function samples random values for each
    component independently.

    .. attention::
        This function applied the same light properties for all the environments.

        position_range is the x, y, z value added into light's cfg.init_pos.
        color_range is the absolute r, g, b value set to the light object.
        intensity_range is the value added into light's cfg.intensity.
        direction_range is the x, y, z value added into light's cfg.direction.
        (Only applicable for ``"sun"``, ``"direction"``, ``"spot"``, ``"rect"``, and ``"mesh"`` light types.)

    .. tip::
        This function uses CPU tensors to assign light properties.

    .. warning::
        ``position_range`` is ignored for global scene lights (``"sun"``, ``"direction"``)
        because they are infinite-distance lights with no meaningful position.
        Use ``direction_range`` instead for these light types.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (Union[torch.Tensor, None]): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        position_range (tuple[list[float], list[float]] | None): The range for the position randomization.
        color_range (tuple[list[float], list[float]] | None): The range for the color randomization.
        intensity_range (tuple[float, float] | None): The range for the intensity randomization.
        direction_range (tuple[list[float], list[float]] | None): The range for the direction randomization.
            Only applicable for directional light types (``"sun"``, ``"direction"``, ``"spot"``,
            ``"rect"``, ``"mesh"``).
    """

    light: Light = env.sim.get_light(entity_cfg.uid)
    if light is None:
        return

    is_global = light.is_global if hasattr(light, "is_global") else False

    # For global lights, normalize env_ids to avoid index-out-of-range
    if is_global or light.num_instances == 1:
        num_instance = 1
        effective_env_ids = None
    else:
        num_instance = len(env_ids)
        effective_env_ids = env_ids

    if position_range:
        if is_global:
            logger.log_warning(
                f"position_range ignored for global light '{entity_cfg.uid}' "
                f"(type='{light.cfg.light_type}'). Use direction_range instead."
            )
        else:
            init_pos = light.cfg.init_pos
            new_pos = (
                torch.tensor(init_pos, dtype=torch.float32)
                .unsqueeze_(0)
                .repeat(num_instance, 1)
            )
            random_value = sample_uniform(
                lower=torch.tensor(position_range[0]),
                upper=torch.tensor(position_range[1]),
                size=new_pos.shape,
            )
            new_pos += random_value
            light.set_local_pose(new_pos, env_ids=effective_env_ids)

    if color_range:
        color = torch.zeros((num_instance, 3), dtype=torch.float32)
        random_value = sample_uniform(
            lower=torch.tensor(color_range[0]),
            upper=torch.tensor(color_range[1]),
            size=color.shape,
        )
        color += random_value
        light.set_color(color, env_ids=effective_env_ids)

    if intensity_range:
        init_intensity = light.cfg.intensity
        new_intensity = (
            torch.tensor(init_intensity, dtype=torch.float32)
            .unsqueeze_(0)
            .repeat(num_instance, 1)
        )
        random_value = sample_uniform(
            lower=torch.tensor(intensity_range[0]),
            upper=torch.tensor(intensity_range[1]),
            size=new_intensity.shape,
        )
        new_intensity += random_value
        new_intensity.squeeze_(1)
        light.set_intensity(new_intensity, env_ids=effective_env_ids)

    if direction_range:
        light_type = light.cfg.light_type
        if light_type not in ("sun", "direction", "spot", "rect", "mesh"):
            logger.log_warning(
                f"direction_range ignored for light '{entity_cfg.uid}' "
                f"(type='{light_type}'). Direction only applicable to "
                f"'sun', 'direction', 'spot', 'rect', and 'mesh' types."
            )
            return

        init_dir = light.cfg.direction
        new_dir = (
            torch.tensor(init_dir, dtype=torch.float32)
            .unsqueeze_(0)
            .repeat(num_instance, 1)
        )
        random_value = sample_uniform(
            lower=torch.tensor(direction_range[0]),
            upper=torch.tensor(direction_range[1]),
            size=new_dir.shape,
        )
        new_dir += random_value
        light.set_direction(new_dir, env_ids=effective_env_ids)


def randomize_emission_light(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    color_range: tuple[list[float], list[float]] | None = None,
    intensity_range: tuple[float, float] | None = None,
) -> None:
    """Randomize emission light properties by adding, scaling, or setting random values.

    This function allows randomizing emission light properties in the scene. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the
    operation.

    The distribution parameters are lists of two elements each, representing the lower and upper bounds of the
    distribution for the r, g, b components of the light color and intensity. The function samples random values for each
    component independently.

    .. attention::
        This function applied the same emission light properties for all the environments.

        color_range is the absolute r, g, b value set on the emission light.
        intensity_range is the absolute intensity value set on the emission light.
    """

    color = None
    if color_range:
        color = torch.zeros((1, 3), dtype=torch.float32)
        random_value = sample_uniform(
            lower=torch.tensor(color_range[0]),
            upper=torch.tensor(color_range[1]),
            size=color.shape,
        )
        color += random_value

    intensity = None
    if intensity_range:
        intensity = np.random.uniform(intensity_range[0], intensity_range[1])

    if isinstance(color, torch.Tensor):
        color_arg = color.squeeze(0).tolist()
    else:
        color_arg = None
    env.sim.set_emission_light(color=color_arg, intensity=intensity)


def randomize_camera_intrinsics(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    focal_x_range: tuple[float, float] | None = None,
    focal_y_range: tuple[float, float] | None = None,
    cx_range: tuple[float, float] | None = None,
    cy_range: tuple[float, float] | None = None,
) -> None:
    """Randomize camera intrinsic properties by adding, scaling, or setting random values.

    This function allows randomizing camera intrinsic parameters in the scene. The function samples random values
    from the given distribution parameters and adds, scales, or sets the values into the physics simulation based
    on the operation.

    The distribution parameters are tuples of two elements each, representing the lower and upper bounds of the
    distribution for the focal length (fx, fy) and principal point (cx, cy) components of the camera intrinsics.
    The function samples random values for each component independently.

    .. attention::
        This function applies the same intrinsic properties for all the environments.

        focal_x_range and focal_y_range are values added to the camera's current fx and fy values.
        focal_xy_range is a combined range for both fx and fy, where the range is specified as
        [[fx_min, fy_min], [fx_max, fy_max]].
        cx_range and cy_range are values added to the camera's current cx and cy values.

    .. tip::
        This function uses CPU tensors to assign camera intrinsic properties.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (Union[torch.Tensor, None]): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        focal_x_range (tuple[float, float] | None): The range for the focal length x randomization.
        focal_y_range (tuple[float, float] | None): The range for the focal length y randomization.
        cx_range (tuple[float, float] | None): The range for the principal point x randomization.
        cy_range (tuple[float, float] | None): The range for the principal point y randomization.
    """

    camera: Union[Camera, StereoCamera] = env.sim.get_sensor(entity_cfg.uid)
    num_instance = len(env_ids)

    # Get current intrinsics as baseline
    current_intrinsics = camera.cfg.intrinsics  # (fx, fy, cx, cy)

    # Create new intrinsics tensor for all instances
    new_intrinsics = (
        torch.tensor(current_intrinsics, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(num_instance, 1)
    )

    # Randomize focal length x (fx)
    if focal_x_range:
        random_value = sample_uniform(
            lower=torch.tensor(focal_x_range[0]),
            upper=torch.tensor(focal_x_range[1]),
            size=(num_instance,),
        )
        new_intrinsics[:, 0] += random_value

    # Randomize focal length y (fy)
    if focal_y_range:
        random_value = sample_uniform(
            lower=torch.tensor(focal_y_range[0]),
            upper=torch.tensor(focal_y_range[1]),
            size=(num_instance,),
        )
        new_intrinsics[:, 1] += random_value

    # Randomize principal point x (cx)
    if cx_range:
        random_value = sample_uniform(
            lower=torch.tensor(cx_range[0]),
            upper=torch.tensor(cx_range[1]),
            size=(num_instance,),
        )
        new_intrinsics[:, 2] += random_value

    # Randomize principal point y (cy)
    if cy_range:
        random_value = sample_uniform(
            lower=torch.tensor(cy_range[0]),
            upper=torch.tensor(cy_range[1]),
            size=(num_instance,),
        )
        new_intrinsics[:, 3] += random_value

    camera.set_intrinsics(new_intrinsics, env_ids=env_ids)


class randomize_visual_material(Functor):
    """Randomize the the visual material properties of a RigidObject or an Articulation.

    Note:
        1. Currently supported randomized properties include:
            - base_color: RGB color of the material. Value should be in [0, 1], shape of (3,)
            - base_color_texture: Texture image for the base color of the material.
                The textures will be preloaded from the given texture_path during initialization.
            - metallic: Metallic property of the material. Value should be in [0, 1].
            - roughness: Roughness property of the material. Value should be in [0, 1].
            - ior: Index of Refraction of the material (only supported in ray tracing mode).
        2. The default ground plane can also be randomized by setting entity_cfg.uid to "default_plane".
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

        self.entity_cfg: SceneEntityCfg = cfg.params["entity_cfg"]

        # special case: default ground plane.
        if self.entity_cfg.uid == "default_plane":
            pass
        else:
            if self.entity_cfg.uid not in env.sim.asset_uids:
                self.entity = None
            else:
                self.entity: Union[RigidObject, Articulation] = env.sim.get_asset(
                    self.entity_cfg.uid
                )

                if not isinstance(self.entity, (RigidObject, Articulation)):
                    raise ValueError(
                        f"Randomization functor 'randomize_visual_material' not supported for asset: '{self.entity_cfg.uid}'"
                        f" with type: '{type(self.entity)}'."
                    )

        # TODO: Maybe need to consider two cases:
        # 1. the texture folder is very large, and we don't want to load all the textures into memory.
        # 2. the texture is generated on the fly.

        # Preload textures (currently only base color textures are supported)
        self.textures = []
        self.texture_refs = []
        texture_path = get_data_path(cfg.params.get("texture_path", None))
        if texture_path is not None:
            from embodichain.utils.utility import read_all_folder_images

            texture_key = str(Path(texture_path).resolve())
            # check if the texture group is already loaded in the global texture cache
            if texture_key in env.sim.get_texture_cache():
                logger.log_info(
                    f"Texture group '{texture_key}' is already loaded in the global texture cache."
                )
                self.textures = env.sim.get_texture_cache(texture_key)
            else:
                self.textures = read_all_folder_images(texture_path)

                # padding the texture with alpha channel if not exist
                for i in range(len(self.textures)):
                    if self.textures[i].shape[2] == 3:
                        data = torch.as_tensor(self.textures[i])
                        alpha_channel = (
                            torch.ones(
                                (data.shape[0], data.shape[1], 1), dtype=data.dtype
                            )
                            * 255
                        )
                        data = torch.cat((data, alpha_channel), dim=2)
                        self.textures[i] = data

                env.sim.set_texture_cache(texture_key, self.textures)

            # Create GPU texture handles once and reuse them on every reset.
            # Passing image data to VisualMaterialInst would recreate a handle
            # for each assignment, which is both expensive and invalidates the
            # intended per-instance reuse semantics.
            self.texture_refs = _get_texture_refs(env.sim, texture_key, self.textures)

        if self.entity_cfg.uid == "default_plane":
            pass

        else:
            # TODO: we may need to get the default material instance from the asset itself.
            mat: VisualMaterial = env.sim.create_visual_material(
                cfg=VisualMaterialCfg(
                    base_color=[1.0, 1.0, 1.0, 1.0],
                    uid=f"{self.entity_cfg.uid}_random_mat",
                )
            )
            if isinstance(self.entity, RigidObject):
                self.entity.set_visual_material(mat)
            elif isinstance(self.entity, Articulation):
                _, link_names = resolve_matching_names(
                    self.entity_cfg.link_names, self.entity.link_names
                )
                self.entity_cfg.link_names = link_names
                self.entity.set_visual_material(mat, link_names=link_names)

        # ground plane only has one instance.
        self._mat_insts = None
        if self.entity_cfg.uid == "default_plane":
            self._mat_insts = env.sim.get_visual_material(
                "plane_mat"
            ).get_default_instance()
            return
        elif isinstance(self.entity, RigidObject):
            self._mat_insts = self.entity.get_visual_material_inst()
            if self.entity.is_shared_visual_material:
                self._mat_insts = self._mat_insts[:1]
        elif isinstance(self.entity, Articulation):
            self._mat_insts = self.entity.get_visual_material_inst(
                link_names=self.entity_cfg.link_names,
            )
            if self.entity.is_shared_visual_material:
                self._mat_insts = self._mat_insts[:1]

    @staticmethod
    def gen_random_base_color_texture(width: int, height: int) -> torch.Tensor:
        """Generate a random base color texture.

        Args:
            width: The width of the texture.
            height: The height of the texture.

        Returns:
            A torch tensor representing the random base color texture with shape (height, width, 4).
        """
        # Generate random RGB values
        rgb = torch.ones((height, width, 3), dtype=torch.float32)
        rgb *= torch.rand((1, 1, 3), dtype=torch.float32)
        rgba = torch.cat((rgb, torch.ones((height, width, 1))), dim=2)
        rgba = (rgba * 255).to(torch.uint8)
        return rgba

    def _randomize_texture(
        self, mat_inst: VisualMaterialInst, texture_idx: int | None = None
    ) -> None:
        if len(self.textures) > 0:
            if texture_idx is None:
                texture_idx = torch.randint(0, len(self.textures), (1,)).item()
            texture_refs = getattr(self, "texture_refs", None)
            if texture_refs:
                mat_inst.set_base_color_texture(texture_ref=texture_refs[texture_idx])
            else:
                mat_inst.set_base_color_texture(texture_data=self.textures[texture_idx])

    def _randomize_mat_inst(
        self,
        mat_inst: VisualMaterialInst,
        plan: Dict[str, torch.Tensor],
        random_texture_prob: float,
        idx: int = 0,
        texture_idx: int | None = None,
    ) -> None:
        # randomize texture or base color based on the probability.
        use_texture = random.random() < random_texture_prob and len(self.textures) != 0
        for key, value in plan.items():
            if key == "base_color":
                mat_inst.set_base_color(value[idx].tolist())
            else:
                getattr(mat_inst, f"set_{key}")(value[idx].item())
        if use_texture:
            self._randomize_texture(mat_inst, texture_idx)
        else:
            # set a random base color instead.
            random_color_texture = (
                randomize_visual_material.gen_random_base_color_texture(2, 2)
            )
            mat_inst.set_base_color_texture(texture_data=random_color_texture)

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        entity_cfg: SceneEntityCfg,
        random_texture_prob: float = 0.5,
        texture_path: str | None = None,
        base_color_range: tuple[list[float], list[float]] | None = None,
        metallic_range: tuple[float, float] | None = None,
        roughness_range: tuple[float, float] | None = None,
        ior_range: tuple[float, float] | None = None,
        texture_sampling: str = "random",
        texture_indices: Mapping[int, int] | None = None,
        texture_scope: str = "per_material",
    ) -> None:
        if self.entity_cfg.uid != "default_plane" and self.entity is None:
            return

        # resolve environment ids
        env_ids = _normalize_env_ids(env_ids, env.num_envs)

        if self.entity_cfg.uid == "default_plane":
            env_ids = [0]

        if texture_scope not in ("per_material", "per_instance"):
            raise ValueError(f"Unsupported texture_scope: {texture_scope}")
        texture_plan = (
            _select_texture_indices(
                texture_sampling, env_ids, len(self.textures), texture_indices
            )
            if self.textures
            else [None] * len(env_ids)
        )

        randomize_plan = {}
        if base_color_range:
            base_color = sample_uniform(
                lower=torch.tensor(base_color_range[0], dtype=torch.float32),
                upper=torch.tensor(base_color_range[1], dtype=torch.float32),
                size=(len(env_ids), 3),  # RGB
            )
            # append alpha channel
            alpha_channel = torch.ones((len(env_ids), 1), dtype=torch.float32)
            base_color = torch.cat((base_color, alpha_channel), dim=1)
            randomize_plan["base_color"] = base_color

        if metallic_range:
            metallic = sample_uniform(
                lower=torch.tensor(metallic_range[0], dtype=torch.float32),
                upper=torch.tensor(metallic_range[1], dtype=torch.float32),
                size=(len(env_ids), 1),
            )
            randomize_plan["metallic"] = metallic

        if roughness_range:
            roughness = sample_uniform(
                lower=torch.tensor(roughness_range[0], dtype=torch.float32),
                upper=torch.tensor(roughness_range[1], dtype=torch.float32),
                size=(len(env_ids), 1),
            )
            randomize_plan["roughness"] = roughness

        if ior_range:
            ior = sample_uniform(
                lower=torch.tensor(ior_range[0], dtype=torch.float32),
                upper=torch.tensor(ior_range[1], dtype=torch.float32),
                size=(len(env_ids), 1),
            )
            randomize_plan["ior"] = ior

        if self.entity_cfg.uid == "default_plane":
            mat_inst = env.sim.get_visual_material("plane_mat").get_default_instance()
            self._randomize_mat_inst(
                mat_inst=mat_inst,
                plan=randomize_plan,
                random_texture_prob=random_texture_prob,
                idx=0,
                texture_idx=texture_plan[0] if texture_plan else None,
            )
            return

        if isinstance(self.entity, RigidObject):
            mat_insts = self.entity.get_visual_material_inst(env_ids=env_ids)
        else:
            mat_insts = self.entity.get_visual_material_inst(
                env_ids=env_ids, link_names=self.entity_cfg.link_names
            )

        for i, data in enumerate(mat_insts):
            if isinstance(self.entity, RigidObject):
                # For RigidObject, data is the material instance directly
                mat: VisualMaterialInst = data
            elif isinstance(self.entity, Articulation):
                # For Articulation, data is the key-value pair of link name and material instance
                mat: Dict[str, VisualMaterialInst] = data

            if isinstance(self.entity, RigidObject):
                self._randomize_mat_inst(
                    mat_inst=mat,
                    plan=randomize_plan,
                    random_texture_prob=random_texture_prob,
                    idx=i,
                    texture_idx=(
                        texture_plan[i]
                        if texture_plan
                        and (
                            texture_scope == "per_instance"
                            or texture_sampling != "random"
                        )
                        else None
                    ),
                )
            else:
                for name, mat_inst in mat.items():
                    self._randomize_mat_inst(
                        mat_inst=mat_inst,
                        plan=randomize_plan,
                        random_texture_prob=random_texture_prob,
                        idx=i,
                        texture_idx=(
                            texture_plan[i]
                            if texture_plan
                            and (
                                texture_scope == "per_instance"
                                or texture_sampling != "random"
                            )
                            else None
                        ),
                    )


class randomize_indirect_lighting(Functor):
    """Randomize the environment's indirect (IBL) lighting or emissive light.

    This functor operates in one of two mutually exclusive modes:

    * **HDR mode** — ``path`` is provided. A random ``.hdr`` file is chosen from
      the folder on every call and applied via :meth:`set_indirect_lighting`.
    * **Emissive mode** — ``emissive_color_range`` and/or
      ``emissive_intensity_range`` are provided. The emissive light color and
      intensity are sampled uniformly on every call and applied via
      :meth:`set_emission_light`.

    Providing both ``path`` and emissive parameters simultaneously is an error.

    .. attention::
        This functor applies the same lighting to all environments.

    .. tip::
        The ``path`` parameter is resolved via :func:`get_data_path`, so it
        supports absolute paths, data-root-relative paths, and dataset-class
        paths (e.g. ``"EnvMapHDR"``).

        ``emissive_color_range`` is a pair of ``[r, g, b]`` lists representing
        the lower and upper bounds for sampling the emissive color, e.g.
        ``[[0.8, 0.8, 0.8], [1.0, 1.0, 1.0]]``.

        ``emissive_intensity_range`` is a ``[min, max]`` pair for the emissive
        intensity scalar, e.g. ``[80.0, 150.0]``.
    """

    def __init__(self, cfg: FunctorCfg, env: EmbodiedEnv):
        """Initialize the functor.

        Args:
            cfg: The configuration of the functor.

                * **HDR mode**: set ``params["path"]`` to a folder of ``.hdr`` files.
                * **Emissive mode**: set ``params["emissive_color_range"]``
                  (pair of RGB lists) and/or ``params["emissive_intensity_range"]``
                  (pair of floats).

            env: The environment instance.

        Raises:
            ValueError: If both HDR and emissive params are provided, or if
                neither is provided.
        """
        super().__init__(cfg, env)

        has_hdr = cfg.params.get("path", None) is not None
        has_emissive = (
            cfg.params.get("emissive_color_range", None) is not None
            or cfg.params.get("emissive_intensity_range", None) is not None
        )

        if has_hdr and has_emissive:
            raise ValueError(
                "randomize_indirect_lighting: 'path' (HDR mode) and emissive "
                "parameters ('emissive_color_range', 'emissive_intensity_range') "
                "are mutually exclusive. Configure only one mode."
            )
        if not has_hdr and not has_emissive:
            raise ValueError(
                "randomize_indirect_lighting: provide either 'path' for HDR "
                "mode, or 'emissive_color_range'/'emissive_intensity_range' for "
                "emissive mode."
            )

        # HDR mode state
        self._hdr_files: list[Path] = []
        if has_hdr:
            path = get_data_path(cfg.params["path"])
            self._hdr_files = sorted(Path(path).glob("*.hdr"))
            if not self._hdr_files:
                logger.log_warning(
                    f"No .hdr files found in '{path}'. "
                    f"Indirect lighting randomization will be a no-op."
                )

        # Emissive mode state
        self._emissive_color_range: tuple[list[float], list[float]] | None = (
            cfg.params.get("emissive_color_range", None)
        )
        self._emissive_intensity_range: tuple[float, float] | None = cfg.params.get(
            "emissive_intensity_range", None
        )

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        path: str | None = None,
    ) -> None:
        """Randomize lighting according to the configured mode.

        In HDR mode a random ``.hdr`` file is selected and applied. In emissive
        mode the emissive color and/or intensity are sampled and applied.

        Args:
            env: The environment instance.
            env_ids: Target environment IDs (unused — lighting is global).
            path: Ignored. Kept for interface compatibility with the event system.
        """
        if self._hdr_files:
            # HDR mode
            selected = random.choice(self._hdr_files)
            env.sim.set_indirect_lighting(str(selected))
            return

        # Emissive mode
        emissive_color: list[float] | None = None
        if self._emissive_color_range is not None:
            color_tensor = sample_uniform(
                lower=torch.tensor(self._emissive_color_range[0]),
                upper=torch.tensor(self._emissive_color_range[1]),
                size=(1, 3),
            )
            emissive_color = color_tensor.squeeze(0).tolist()

        emissive_intensity: float | None = None
        if self._emissive_intensity_range is not None:
            emissive_intensity = float(
                np.random.uniform(
                    self._emissive_intensity_range[0],
                    self._emissive_intensity_range[1],
                )
            )

        env.sim.set_emission_light(color=emissive_color, intensity=emissive_intensity)
