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
from typing import TYPE_CHECKING, Literal, Union, Optional, Dict

from embodichain.lab.sim.objects import Light, RigidObject, Articulation
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
    "randomize_camera_intrinsics",
    "randomize_visual_material",
]


def randomize_camera_extrinsics(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    pos_range: Optional[tuple[list[float], list[float]]] = None,
    euler_range: Optional[tuple[list[float], list[float]]] = None,
    eye_range: Optional[tuple[list[float], list[float]]] = None,
    target_range: Optional[tuple[list[float], list[float]]] = None,
    up_range: Optional[tuple[list[float], list[float]]] = None,
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
            init_euler = euler_xyz_from_quat(init_quat_np)
            # 2. Sample perturbation for euler angles
            random_value = sample_uniform(
                lower=torch.tensor(euler_range[0]),
                upper=torch.tensor(euler_range[1]),
                size=(num_instance, 3),
            )
            # 3. Add perturbation to each environment and convert back to quaternion
            new_quat = quat_from_euler_xyz(init_euler + random_value)
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
    position_range: Optional[tuple[list[float], list[float]]] = None,
    color_range: Optional[tuple[list[float], list[float]]] = None,
    intensity_range: Optional[tuple[float, float]] = None,
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

    .. tip::
        This function uses CPU tensors to assign light properties.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (Union[torch.Tensor, None]): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration of the scene entity to randomize.
        position_range (Optional[tuple[list[float], list[float]]]): The range for the position randomization.
        color_range (Optional[tuple[list[float], list[float]]]): The range for the color randomization.
        intensity_range (Optional[tuple[float, float]]): The range for the intensity randomization.
    """

    light: Light = env.sim.get_light(entity_cfg.uid)
    num_instance = len(env_ids)

    if position_range:
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
        light.set_local_pose(new_pos, env_ids=env_ids)

    if color_range:
        color = torch.zeros((num_instance, 3), dtype=torch.float32)
        random_value = sample_uniform(
            lower=torch.tensor(color_range[0]),
            upper=torch.tensor(color_range[1]),
            size=color.shape,
        )
        color += random_value
        light.set_color(color, env_ids=env_ids)

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
        light.set_intensity(new_intensity, env_ids=env_ids)


def randomize_camera_intrinsics(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    focal_x_range: Optional[tuple[float, float]] = None,
    focal_y_range: Optional[tuple[float, float]] = None,
    cx_range: Optional[tuple[float, float]] = None,
    cy_range: Optional[tuple[float, float]] = None,
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
        focal_x_range (Optional[tuple[float, float]]): The range for the focal length x randomization.
        focal_y_range (Optional[tuple[float, float]]): The range for the focal length y randomization.
        cx_range (Optional[tuple[float, float]]): The range for the principal point x randomization.
        cy_range (Optional[tuple[float, float]]): The range for the principal point y randomization.
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
        texture_path = get_data_path(cfg.params.get("texture_path", None))
        if texture_path is not None:
            from embodichain.utils.utility import read_all_folder_images

            texture_key = os.path.basename(texture_path)
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

    def _randomize_texture(self, mat_inst: VisualMaterialInst) -> None:
        if len(self.textures) > 0:
            # Randomly select a texture from the preloaded textures
            texture_idx = torch.randint(0, len(self.textures), (1,)).item()
            mat_inst.set_base_color_texture(texture_data=self.textures[texture_idx])

    def _randomize_mat_inst(
        self,
        mat_inst: VisualMaterialInst,
        plan: Dict[str, torch.Tensor],
        random_texture_prob: float,
        idx: int = 0,
    ) -> None:
        # randomize the material instance pbr properties based on the plan.
        for key, value in plan.items():
            if key == "base_color":
                mat_inst.set_base_color(value[idx].tolist())
            else:
                getattr(mat_inst, f"set_{key}")(value[idx].item())

        # randomize texture or base color based on the probability.
        if random_texture_prob <= 0.0 or len(self.textures) == 0:
            return
        if random.random() < random_texture_prob:
            self._randomize_texture(mat_inst)
        else:
            # set a random base color instead.
            random_color = torch.rand(3).tolist()
            random_color.append(1.0)  # alpha
            mat_inst.set_base_color(random_color)

    def __call__(
        self,
        env: EmbodiedEnv,
        env_ids: Union[torch.Tensor, None],
        entity_cfg: SceneEntityCfg,
        random_texture_prob: float = 0.5,
        texture_path: Optional[str] = None,
        base_color_range: Optional[tuple[list[float], list[float]]] = None,
        metallic_range: Optional[tuple[float, float]] = None,
        roughness_range: Optional[tuple[float, float]] = None,
        ior_range: Optional[tuple[float, float]] = None,
    ):
        from embodichain.lab.sim.utility import is_rt_enabled

        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        if self.entity_cfg.uid == "default_plane":
            env_ids = [0]

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

        if ior_range and is_rt_enabled():
            ior = sample_uniform(
                lower=torch.tensor(ior_range[0], dtype=torch.float32),
                upper=torch.tensor(ior_range[1], dtype=torch.float32),
                size=(len(env_ids), 1),
            )
            randomize_plan["ior"] = ior

        # ground plane only has one instance.
        mat_insts = None
        if self.entity_cfg.uid == "default_plane":
            mat_inst = env.sim.get_visual_material("plane_mat").get_default_instance()
            self._randomize_mat_inst(
                mat_inst=mat_inst,
                plan=randomize_plan,
                random_texture_prob=random_texture_prob,
                idx=0,
            )
            return
        elif isinstance(self.entity, RigidObject):
            mat_insts = self.entity.get_visual_material_inst(env_ids=env_ids)
        elif isinstance(self.entity, Articulation):
            mat_insts = self.entity.get_visual_material_inst(
                env_ids=env_ids,
                link_names=self.entity_cfg.link_names,
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
                )
            else:
                for name, mat_inst in mat.items():
                    self._randomize_mat_inst(
                        mat_inst=mat_inst,
                        plan=randomize_plan,
                        random_texture_prob=random_texture_prob,
                        idx=i,
                    )
