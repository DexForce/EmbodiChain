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

from typing import TYPE_CHECKING, List, Union

import torch

from embodichain.lab.gym.envs.managers.cfg import SceneEntityCfg
from embodichain.lab.sim.objects import RigidObject
from embodichain.utils import logger
from embodichain.utils.math import sample_uniform

if TYPE_CHECKING:
    from embodichain.lab.gym.envs import EmbodiedEnv


def _normalize_env_ids(
    env: EmbodiedEnv, env_ids: Union[torch.Tensor, None]
) -> torch.Tensor:
    # Target all active environments if no specific IDs are provided
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    return env_ids


def _sample_body_scale(
    env: EmbodiedEnv,
    env_ids: torch.Tensor,
    scale_factor_range: tuple[list[float], list[float]],
    same_scale_all_axes: bool,
) -> torch.Tensor:
    """Sample per-env body scale factors.

    Returns:
        torch.Tensor: Shape (num_envs_selected, 3) scale factors for x/y/z.
    """
    num_instance = len(env_ids)
    if same_scale_all_axes:
        low = torch.tensor(scale_factor_range[0][0], device=env.device)
        high = torch.tensor(scale_factor_range[1][0], device=env.device)
        s = sample_uniform(lower=low, upper=high, size=(num_instance,))
        return torch.stack([s, s, s], dim=1)
    low = torch.tensor(scale_factor_range[0], device=env.device)
    high = torch.tensor(scale_factor_range[1], device=env.device)
    return sample_uniform(lower=low, upper=high, size=(num_instance, 3))


def randomize_rigid_object_scale(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    scale_factor_range: tuple[list[float], list[float]] | None = None,
    same_scale_all_axes: bool = True,
) -> None:
    """Randomize a rigid object's *body scale factors* (multiplicative, not absolute size).

    Args:
        env: Environment instance.
        env_ids: Target env ids. If None, applies to all envs.
        entity_cfg: Scene entity config of the rigid object.
        scale_factor_range: If same_scale_all_axes is True, should be [[s_min], [s_max]].
            Otherwise [[sx_min, sy_min, sz_min], [sx_max, sy_max, sz_max]].
        same_scale_all_axes: Whether to use same factor on x/y/z.
    """
    if scale_factor_range is None:
        return
    if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
        return

    env_ids = _normalize_env_ids(env, env_ids)
    rigid_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)
    scale = _sample_body_scale(env, env_ids, scale_factor_range, same_scale_all_axes)
    rigid_object.set_body_scale(scale, env_ids=env_ids)


def randomize_rigid_objects_scale(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfgs: List[SceneEntityCfg],
    scale_factor_range: tuple[list[float], list[float]] | None = None,
    same_scale_all_axes: bool = True,
    shared_sample: bool = False,
) -> None:
    """Randomize body scale factors for multiple rigid objects.

    Args:
        env: Environment instance.
        env_ids: Target env ids. If None, applies to all envs.
        entity_cfgs: List of scene entity configs (rigid objects).
        scale_factor_range: Scale factor sampling range.
        same_scale_all_axes: Whether to use same factor on x/y/z.
        shared_sample: If True, sample one scale per-env and apply to *all* objects (sync).
            If False, each object samples its own scales independently.
    """
    if scale_factor_range is None:
        return

    if not isinstance(entity_cfgs, list) or len(entity_cfgs) == 0:
        return

    env_ids = _normalize_env_ids(env, env_ids)

    if shared_sample:
        scale = _sample_body_scale(
            env, env_ids, scale_factor_range, same_scale_all_axes
        )
        for entity_cfg in entity_cfgs:
            if entity_cfg.uid not in env.sim.get_rigid_object_uid_list():
                continue
            rigid_object: RigidObject = env.sim.get_rigid_object(entity_cfg.uid)
            rigid_object.set_body_scale(scale, env_ids=env_ids)
        return

    for entity_cfg in entity_cfgs:
        randomize_rigid_object_scale(
            env=env,
            env_ids=env_ids,
            entity_cfg=entity_cfg,
            scale_factor_range=scale_factor_range,
            same_scale_all_axes=same_scale_all_axes,
        )


def randomize_rigid_object_body_scale(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfg: SceneEntityCfg,
    scale_range: tuple[list[float], list[float]] | None = None,
    same_scale_all_axes: bool = True,
) -> None:
    """Deprecated. Use `randomize_rigid_object_scale` + `scale_factor_range`."""
    if scale_range is not None:
        logger.log_warning(
            "`randomize_rigid_object_body_scale` is deprecated. "
            "Please migrate to `randomize_rigid_object_scale` with `scale_factor_range`."
        )
    return randomize_rigid_object_scale(
        env=env,
        env_ids=env_ids,
        entity_cfg=entity_cfg,
        scale_factor_range=scale_range,
        same_scale_all_axes=same_scale_all_axes,
    )


def randomize_rigid_objects_body_scale_sync(
    env: EmbodiedEnv,
    env_ids: Union[torch.Tensor, None],
    entity_cfgs: List[SceneEntityCfg],
    scale_range: tuple[list[float], list[float]] | None = None,
    same_scale_all_axes: bool = True,
) -> None:
    """Deprecated. Use `randomize_rigid_objects_scale(shared_sample=True)` + `scale_factor_range`."""
    if scale_range is not None:
        logger.log_warning(
            "`randomize_rigid_objects_body_scale_sync` is deprecated. "
            "Please migrate to `randomize_rigid_objects_scale` with `shared_sample=true` "
            "and `scale_factor_range`."
        )
    return randomize_rigid_objects_scale(
        env=env,
        env_ids=env_ids,
        entity_cfgs=entity_cfgs,
        scale_factor_range=scale_range,
        same_scale_all_axes=same_scale_all_axes,
        shared_sample=True,
    )
