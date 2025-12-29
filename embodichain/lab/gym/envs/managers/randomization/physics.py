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
from embodichain.utils.math import sample_uniform


if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


def randomize_rigid_object_mass(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, List[int]],
    entity_cfg: SceneEntityCfg,
    mass_range: tuple[float, float],
    relative: bool = False,
) -> None:
    """Randomize the mass of rigid objects in the environment.

    Args:
        env (EmbodiedEnv): The environment instance.
        env_ids (Union[torch.Tensor, List[int]]): The environment IDs to apply the randomization.
        entity_cfg (SceneEntityCfg): The configuration for the scene entity.
        mass_range (tuple[float, float]): The range (min, max) to sample the mass from.
        relative (bool): Whether to apply the mass change relative to the initial mass. Defaults to False.
    """

    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return

    rigid_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)
    num_instance = len(env_ids)

    sampled_masses = sample_uniform(
        lower=mass_range[0], upper=mass_range[1], size=(num_instance,)
    )

    if relative:
        init_mass = rigid_object.cfg.attrs.mass
        init_mass = torch.full((sampled_masses.shape), init_mass, device=env.device)
        sampled_masses = init_mass + sampled_masses

    rigid_object.set_mass(sampled_masses, env_ids=env_ids)
