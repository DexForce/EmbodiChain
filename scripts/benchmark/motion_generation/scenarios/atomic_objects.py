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

"""Config-driven rigid objects for Atomic Task benchmark cases."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import RigidObject

__all__ = [
    "AtomicObjectHandle",
    "atomic_object_kind_names",
    "create_atomic_object",
    "register_atomic_object_kind",
]

AtomicShapeFactory = Callable[[Mapping[str, object]], object]
_ATOMIC_OBJECT_KINDS: dict[str, AtomicShapeFactory] = {}


@dataclass
class AtomicObjectHandle:
    """Simulation object plus its algorithm-independent frozen state."""

    object_id: str
    kind: str
    config: dict[str, object]
    entity: "RigidObject"
    initial_pose: torch.Tensor

    def reset(self) -> None:
        """Restore the frozen initial pose and clear residual dynamics."""
        self.entity.set_local_pose(self.initial_pose)
        self.entity.clear_dynamics()

    def park(self, index: int) -> None:
        """Move an inactive object outside every benchmark workspace."""
        pose = self.initial_pose.clone()
        pose[:, 0, 3] = 8.0 + float(index)
        pose[:, 1, 3] = 8.0
        pose[:, 2, 3] = 1.0
        self.entity.set_local_pose(pose)
        self.entity.clear_dynamics()


def register_atomic_object_kind(name: str, factory: AtomicShapeFactory) -> None:
    """Register a config-only object shape factory."""
    if not name:
        raise ValueError("Atomic object kind must not be empty.")
    previous = _ATOMIC_OBJECT_KINDS.get(name)
    if previous is not None and previous is not factory:
        raise ValueError(f"Atomic object kind {name!r} is already registered.")
    _ATOMIC_OBJECT_KINDS[name] = factory


def atomic_object_kind_names() -> tuple[str, ...]:
    """Return registered object kinds in deterministic order."""
    return tuple(sorted(_ATOMIC_OBJECT_KINDS))


def _vector(
    value: object, *, name: str, length: int, default: Sequence[float]
) -> list[float]:
    """Validate and normalize a numeric vector from YAML configuration."""
    resolved = default if value is None else value
    if not isinstance(resolved, Sequence) or isinstance(resolved, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of {length} numbers.")
    result = [float(item) for item in resolved]
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values.")
    return result


def _cube_shape(config: Mapping[str, object]) -> CubeCfg:
    """Build a cube shape from its declarative size."""
    size = _vector(
        config.get("size"), name="cube.size", length=3, default=(0.05, 0.05, 0.05)
    )
    if any(value <= 0.0 for value in size):
        raise ValueError("cube.size values must be greater than zero.")
    return CubeCfg(size=size)


def _mesh_shape(config: Mapping[str, object]) -> MeshCfg:
    """Build a mesh shape from an absolute or EmbodiChain data path."""
    asset_path = config.get("asset_path")
    if not isinstance(asset_path, str) or not asset_path:
        raise ValueError("mesh.asset_path must be a non-empty string.")
    resolved = Path(asset_path)
    return MeshCfg(
        fpath=str(resolved if resolved.is_absolute() else get_data_path(asset_path))
    )


def create_atomic_object(
    simulation: "SimulationManager", object_config: Mapping[str, object]
) -> AtomicObjectHandle:
    """Create one config-driven object and freeze its settled initial pose."""
    object_id = object_config.get("id")
    kind = object_config.get("kind")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Every atomic object must define a non-empty id.")
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"Atomic object {object_id!r} must define a kind.")
    try:
        factory = _ATOMIC_OBJECT_KINDS[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown atomic object kind {kind!r}; registered kinds: "
            f"{atomic_object_kind_names()}."
        ) from exc

    config = dict(object_config)
    position = _vector(
        config.get("position"),
        name=f"objects[{object_id}].position",
        length=3,
        default=(-0.42, -0.08, 0.05),
    )
    rotation = _vector(
        config.get("rotation_deg"),
        name=f"objects[{object_id}].rotation_deg",
        length=3,
        default=(0.0, 0.0, 0.0),
    )
    scale = _vector(
        config.get("scale"),
        name=f"objects[{object_id}].scale",
        length=3,
        default=(1.0, 1.0, 1.0),
    )
    entity = simulation.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=f"atomic_benchmark_{object_id}",
            shape=factory(config),
            attrs=RigidBodyAttributesCfg(
                mass=float(config.get("mass", 0.05)),
                dynamic_friction=float(config.get("dynamic_friction", 0.97)),
                static_friction=float(config.get("static_friction", 0.99)),
                restitution=float(config.get("restitution", 0.0)),
                contact_offset=float(config.get("contact_offset", 0.003)),
                rest_offset=float(config.get("rest_offset", 0.001)),
                linear_damping=float(config.get("linear_damping", 0.7)),
                angular_damping=float(config.get("angular_damping", 0.7)),
                min_position_iters=int(config.get("min_position_iters", 32)),
                min_velocity_iters=int(config.get("min_velocity_iters", 8)),
            ),
            max_convex_hull_num=int(config.get("max_convex_hull_num", 16)),
            init_pos=position,
            init_rot=rotation,
            body_scale=scale,
            use_usd_properties=bool(config.get("use_usd_properties", False)),
        )
    )
    simulation.update(step=int(config.get("settle_steps", 10)))
    entity.clear_dynamics()
    return AtomicObjectHandle(
        object_id=object_id,
        kind=kind,
        config=config,
        entity=entity,
        initial_pose=entity.get_local_pose(to_matrix=True).clone(),
    )


register_atomic_object_kind("cube", _cube_shape)
register_atomic_object_kind("mesh", _mesh_shape)
