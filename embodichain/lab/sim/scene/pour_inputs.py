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

"""Exact physical inputs for source-compatible Pour staging services."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import torch
from embodichain.compute.geometry.pouring import _pose
from .physical_parameters import observed_gravity

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot, RigidObject
    from embodichain.lab.sim.sim_manager import SimulationManager
__all__ = [
    "capture_pour_vertices",
    "capture_pour_faces",
    "configured_pour_gravity",
    "observed_pour_arm_origin",
]


def _readonly(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result.setflags(write=False)
    return result


def capture_pour_vertices(entity: RigidObject) -> np.ndarray:
    """Read one B1 native body-local mesh, applying body scale exactly once.

    Args:
        entity: Exact bound native rigid object with one instance.

    Returns:
        Owned read-only local vertex array retaining the native dtype/order.
    """
    getter = getattr(entity, "get_vertices", None)
    if not callable(getter):
        raise TypeError("Pour geometry requires a native rigid-object state provider.")
    if getattr(entity, "num_instances", None) != 1:
        raise ValueError("Pour native geometry currently requires one native instance.")
    vertices = getter(env_ids=[0], scale=True)
    if (
        not isinstance(vertices, torch.Tensor)
        or vertices.ndim != 3
        or vertices.shape[0] != 1
        or vertices.shape[1] == 0
        or vertices.shape[2] != 3
        or not vertices.is_floating_point()
        or not torch.isfinite(vertices).all()
    ):
        raise ValueError("Pour native vertices must be finite floating (1, N, 3).")
    return _readonly(vertices[0].detach().cpu().numpy())


def configured_pour_gravity(
    simulation: SimulationManager,
) -> tuple[float, float, float]:
    """Read the actual Default/B1 configuration; never invent missing gravity.

    Args:
        simulation: Exact physical owner with explicit physics configuration.

    Returns:
        Finite nonzero configured world gravity. Unsupported backend/batch fails.
    """
    if (
        getattr(simulation, "physics_backend", None) != "default"
        or getattr(simulation, "num_envs", None) != 1
    ):
        raise ValueError("Explicit pour services currently require Default/B1.")
    try:
        return observed_gravity(simulation)
    except ValueError as error:
        raise ValueError(
            "Pour services require a finite nonzero configured gravity."
        ) from error


def observed_pour_arm_origin(
    robot: Robot, control_part: str, env_ids: torch.Tensor
) -> np.ndarray:
    """Read the compiled arm solver's current root, never guess the robot base.

    Args:
        robot: Exact assembled robot owning the selected solver.
        control_part: Explicit active motion control group.
        env_ids: Exact B1 native environment selection.

    Returns:
        Owned finite world position of the configured arm root.
    """
    if env_ids.shape != (1,):
        raise ValueError("Pour region arm observation requires an exact B1 row.")
    solver = robot.get_solver(control_part)
    name = getattr(solver, "root_link_name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("Pour region requires the bound arm solver's root link.")
    pose = robot.get_link_pose(link_name=name, env_ids=env_ids, to_matrix=True)
    if not isinstance(pose, torch.Tensor) or pose.shape != (1, 4, 4):
        raise ValueError("Pour region requires an observed B1 native arm-root pose.")

    return _pose(pose[0].detach().cpu().numpy())[:3, 3].copy()


def capture_pour_faces(entity: RigidObject) -> np.ndarray:
    """Capture exact native B1 receiver triangles without poses or topology repair.

    Args:
        entity: Exact bound native receiver object.

    Returns:
        Owned CPU integer face rows retaining the original order.
    """
    if getattr(entity, "num_instances", None) != 1:
        raise ValueError("Pour receiver faces require one native instance.")
    getter = getattr(entity, "get_triangles", None)
    if not callable(getter):
        raise TypeError("Pour region requires native receiver triangles.")
    value = getter(env_ids=[0])
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or value.shape[0] != 1
        or value.shape[2] != 3
        or value.dtype == torch.bool
        or value.is_floating_point()
        or value.is_complex()
    ):
        raise ValueError("Pour native receiver faces must be integer (1,F,3).")
    return value[0].detach().cpu().numpy().copy()
