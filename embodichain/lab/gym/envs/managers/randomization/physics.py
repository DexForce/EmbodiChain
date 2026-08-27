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
from typing import TYPE_CHECKING

from embodichain.lab.sim.objects import Articulation, RigidObject, Robot
from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.utils.math import sample_uniform
from embodichain.utils.string import resolve_matching_names
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


def randomize_rigid_object_mass(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | list[int],
    entity_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
    relative: bool = False,
    recompute_inertia: bool = True,
    min_mass: float = 1e-6,
) -> None:
    """Randomize the mass of rigid objects in the environment.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | list[int]): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration for the scene entity.
        mass_range (tuple[float, float]): The range (min, max) to sample the mass from.
        relative (bool): Whether to apply the mass change relative to the initial mass. Defaults to False.
        recompute_inertia (bool): Whether to scale the initial inertia by the sampled
            mass ratio. Defaults to True.
        min_mass (float): Minimum allowed sampled mass. Defaults to 1e-6.

    Raises:
        ValueError: If ``min_mass`` is not positive or an initial mass is not positive.
    """

    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return

    rigid_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)
    if rigid_object.is_non_dynamic:
        logger.log_warning(
            f"Cannot randomize mass for non-dynamic rigid object '{entity_cfg.uid}'."
        )
        return
    if min_mass <= 0.0:
        raise ValueError(f"min_mass must be positive, got {min_mass}.")

    num_instance = len(env_ids)
    index = torch.as_tensor(env_ids, dtype=torch.long, device=rigid_object.device)
    body_data = rigid_object.body_data
    if body_data is None:
        return
    default_masses = body_data.default_mass[index]
    if torch.any(default_masses <= 0.0):
        raise ValueError("Initial rigid-body masses must be positive.")

    sampled_masses = sample_uniform(
        lower=mass_range[0],
        upper=mass_range[1],
        size=(num_instance,),
        device=rigid_object.device,
    )

    if relative:
        sampled_masses = default_masses + sampled_masses

    sampled_masses = sampled_masses.clamp_min(min_mass)

    rigid_object.set_mass(sampled_masses, env_ids=env_ids)

    if recompute_inertia:
        mass_ratios = sampled_masses / default_masses
        sampled_inertia = body_data.default_inertia[index] * mass_ratios.unsqueeze(-1)
        rigid_object.set_inertia(sampled_inertia, env_ids=env_ids)


def randomize_rigid_object_center_of_mass(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | list[int],
    entity_cfg: SceneEntityCfg,
    com_pos_offset_range: tuple[list[float], list[float]],
) -> None:
    """Randomize the center of mass of rigid objects in the environment.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | list[int]): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration for the scene entity.
        com_pos_offset_range (tuple[list[float], list[float]]): The range (min, max) to sample the center of mass offset from.
    """

    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return

    rigid_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)
    if rigid_object.is_non_dynamic:
        logger.log_warning(
            f"Cannot randomize center of mass for non-dynamic rigid object '{entity_cfg.uid}'."
        )
        return

    num_instance = len(env_ids)

    sampled_com_pos_offsets = sample_uniform(
        lower=com_pos_offset_range[0],
        upper=com_pos_offset_range[1],
        size=(num_instance, 3),
    )

    com = rigid_object.body_data.default_com_pose[env_ids]
    updated_com = com.clone()
    updated_com[:, 0:3] += sampled_com_pos_offsets

    rigid_object.set_com_pose(updated_com, env_ids=env_ids)


def randomize_articulation_mass(
    env: EmbodiedEnv,
    env_ids: torch.Tensor | list[int],
    entity_cfg: SceneEntityCfg,
    mass_range: tuple[float, float] | dict[str, tuple[float, float]],
    link_names: str | list[str] | None = None,
    relative: bool = False,
    recompute_inertia: bool = True,
    min_mass: float = 1e-6,
) -> None:
    """Randomize the mass of articulation links in the environment.

    Uses regular expression matching to select which links to randomize.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (torch.Tensor | list[int]): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration for the scene entity.
        mass_range (tuple[float, float] | dict[str, tuple[float, float]]): The range (min, max)
            to sample the mass from. If a dict, keys are link names and values are per-link
            mass ranges. When a dict is provided, ``link_names`` is ignored and the dict keys
            are used instead.
        link_names (str | list[str] | None): A regex pattern or list of regex patterns to match
            link names. If None, all links are randomized. Ignored when ``mass_range`` is a dict.
            Defaults to None.
        relative (bool): Whether to apply the mass change relative to the initial mass.
            Defaults to False.
        recompute_inertia (bool): Whether to scale initialization-time inertia by
            the sampled mass ratio. Defaults to True.
        min_mass (float): Minimum allowed sampled mass. Defaults to 1e-6.

    Raises:
        ValueError: If ``min_mass`` or an initialization-time link mass is not
            positive.
    """

    if entity_cfg.uid not in env.sim.get_articulation_uid_list():
        return

    articulation: Articulation = env.sim.get_articulation(entity_cfg.uid)
    if min_mass <= 0.0:
        raise ValueError(f"min_mass must be positive, got {min_mass}.")
    num_instance = len(env_ids)

    if isinstance(mass_range, dict):
        # Check the link names in the dict keys are valid.
        for name in mass_range.keys():
            if name not in articulation.link_names:
                raise ValueError(
                    f"Link name '{name}' in mass_range dict is not found in articulation '{entity_cfg.uid}'."
                )

        # Per-link mass ranges: dict keys are exact link names
        matched_link_names = list(mass_range.keys())
        link_lower = torch.tensor(
            [mass_range[name][0] for name in matched_link_names],
            device=articulation.device,
            dtype=torch.float32,
        )
        link_upper = torch.tensor(
            [mass_range[name][1] for name in matched_link_names],
            device=articulation.device,
            dtype=torch.float32,
        )
        # Broadcast: (num_instance, num_links)
        sampled_masses = torch.rand(
            (num_instance, len(matched_link_names)),
            device=articulation.device,
            dtype=torch.float32,
        )
        sampled_masses = link_lower + sampled_masses * (link_upper - link_lower)
    else:
        # Resolve link names via regex matching
        if link_names is not None:
            _, matched_link_names = resolve_matching_names(
                keys=link_names,
                list_of_strings=articulation.link_names,
            )
        else:
            matched_link_names = articulation.link_names

        # Sample masses: shape (num_instance, num_links)
        sampled_masses = sample_uniform(
            lower=mass_range[0],
            upper=mass_range[1],
            size=(num_instance, len(matched_link_names)),
            device=articulation.device,
        )

    env_index = torch.as_tensor(env_ids, dtype=torch.long, device=articulation.device)
    link_indices = torch.as_tensor(
        [articulation.link_names.index(name) for name in matched_link_names],
        dtype=torch.long,
        device=articulation.device,
    )
    default_masses = articulation.body_data.default_mass[
        env_index[:, None], link_indices[None, :]
    ]
    if torch.any(default_masses <= 0.0):
        raise ValueError(
            "Initialization-time articulation link masses must be positive."
        )

    if relative:
        sampled_masses = default_masses + sampled_masses

    sampled_masses = sampled_masses.clamp_min(min_mass)

    articulation.set_mass(
        sampled_masses, link_names=matched_link_names, env_ids=env_ids
    )

    if recompute_inertia:
        default_inertia = articulation.body_data.default_inertia[
            env_index[:, None], link_indices[None, :]
        ]
        mass_ratios = sampled_masses / default_masses
        articulation.set_inertia(
            default_inertia * mass_ratios.unsqueeze(-1),
            link_names=matched_link_names,
            env_ids=env_ids,
        )
