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
"""Backend-specific native entity lifetime helpers."""

from __future__ import annotations

from typing import Sequence

import torch

from embodichain.lab.sim.physics.newton import is_newton_gradient_mode

from .newton import is_newton_scene

__all__ = [
    "destroy_articulation_entities",
    "destroy_rigid_entities",
    "finalize_articulation_spawn",
    "apply_rigid_initial_state",
]


def _world_arenas(world: object) -> list[object]:
    env = world.get_env()
    arenas = env.get_all_arenas()
    return arenas if len(arenas) else [env]


def destroy_rigid_entities(
    world: object,
    scene: object,
    entities: Sequence[object],
    *,
    is_newton: bool | None = None,
) -> None:
    """Remove unbound rigid entities using the active backend's API."""
    arenas = _world_arenas(world)
    newton = is_newton_scene(scene) if is_newton is None else is_newton
    for index, entity in enumerate(entities):
        if newton:
            arenas[index].remove_actor(entity.get_name())
        else:
            arenas[index].remove_actor(entity)


def destroy_articulation_entities(
    world: object,
    scene: object,
    entities: Sequence[object],
    *,
    is_newton: bool | None = None,
) -> None:
    """Remove unbound articulation entities using the active backend's API."""
    arenas = _world_arenas(world)
    newton = is_newton_scene(scene) if is_newton is None else is_newton
    for index, entity in enumerate(entities):
        if newton:
            arenas[index].remove_skeleton(entity)
        else:
            arenas[index].remove_articulation(entity)


def finalize_articulation_spawn(articulation: object, result: object) -> None:
    """Apply backend-specific initial-state handling after Spawn binding."""
    if is_newton_gradient_mode(result):
        initial_qpos = torch.as_tensor(articulation.cfg.init_qpos).reshape(-1)
        if initial_qpos.numel() != articulation.dof:
            raise ValueError(
                f"Articulation {articulation.uid!r} expected {articulation.dof} "
                f"initial joint positions, got {initial_qpos.numel()}."
            )
        if torch.any(initial_qpos != 0.0):
            raise NotImplementedError(
                "Newton gradient mode cannot apply non-zero init_qpos after "
                "Spawn finalization. Author the initial coordinates in the "
                "source asset or initialize them in a differentiable task "
                "before opening a Warp tape."
            )
        return

    if articulation._data.is_newton_backend:
        # Newton finalization already authored the initial state. Clearing one
        # facade would select only part of a multi-articulation world.
        articulation.reset(clear_dynamics=False)
    else:
        articulation.reset()


def apply_rigid_initial_state(rigid_object: object) -> None:
    """Apply backend-specific initial-state handling after rigid construction."""
    if rigid_object.is_spawn_bound:
        if rigid_object._spawn_result.backend == "dexsim":
            rigid_object.reset()
        elif not is_newton_gradient_mode(rigid_object._spawn_result):
            rigid_object.clear_dynamics()
        return

    if is_newton_scene(rigid_object._ps):
        if rigid_object._newton_lifecycle_state() == "BUILDER":
            rigid_object.set_local_pose(
                rigid_object._build_cfg_init_pose(rigid_object._all_indices),
                env_ids=rigid_object._all_indices,
            )
        return

    if rigid_object.device.type == "cuda":
        rigid_object._world.update(0.001)
    rigid_object.reset()
