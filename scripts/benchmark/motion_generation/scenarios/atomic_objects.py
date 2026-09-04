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

"""Config-driven scene objects for Atomic Task benchmark cases."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.utils.math import matrix_from_euler

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.objects import Articulation, RigidObject

__all__ = [
    "AtomicArticulationHandle",
    "AtomicObjectHandle",
    "atomic_object_kind_names",
    "create_atomic_articulation",
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


@dataclass
class AtomicArticulationHandle:
    """Simulation articulation plus its frozen benchmark reset state."""

    object_id: str
    config: dict[str, object]
    entity: "Articulation"
    initial_pose: torch.Tensor
    initial_qpos: torch.Tensor

    def reset(
        self,
        *,
        pose: torch.Tensor | None = None,
        qpos: torch.Tensor | None = None,
    ) -> None:
        """Restore root pose, actual joint state, drive target, and dynamics."""
        resolved_pose = self.initial_pose if pose is None else pose
        resolved_qpos = self.initial_qpos if qpos is None else qpos
        self.entity.set_qpos(resolved_qpos, target=False)
        self.entity.set_qpos(resolved_qpos, target=True)
        # On CUDA, set_qpos writes the articulation state but does not refresh
        # link transforms.  Applying the root pose last runs the required
        # articulation kinematics update for the new qpos.
        self.entity.set_local_pose(resolved_pose)
        self.entity.clear_dynamics()

    def park(self, index: int) -> None:
        """Reset and move an inactive articulation outside the workspace."""
        pose = self.initial_pose.clone()
        pose[:, 0, 3] = 8.0 + float(index)
        pose[:, 1, 3] = 8.0
        pose[:, 2, 3] = 1.0
        self.reset(pose=pose)

    def link_pose(self, link_name: str) -> torch.Tensor:
        """Return a cloned world pose for one articulation link."""
        return self.entity.get_link_pose(link_name, to_matrix=True).clone()

    def link_mesh(self, link_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cloned target-local mesh tensors for one articulation link."""
        vertices, triangles = self.entity.get_link_vert_face(link_name)
        return vertices.clone(), triangles.clone()

    def joint_qpos(self, joint_name: str) -> torch.Tensor:
        """Return the live position of one named active joint."""
        try:
            joint_id = self.entity.joint_names.index(joint_name)
        except ValueError as exc:
            raise ValueError(
                f"Unknown joint {joint_name!r} on articulation {self.object_id!r}; "
                f"available joints: {list(self.entity.joint_names)}."
            ) from exc
        return self.entity.get_qpos()[:, joint_id].clone()


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


def _configured_pose(
    simulation: "SimulationManager",
    position: Sequence[float],
    rotation_deg: Sequence[float],
) -> torch.Tensor:
    """Build the exact batched root pose declared by one scene entry."""
    position_tensor = torch.tensor(
        position, dtype=torch.float32, device=simulation.device
    ).repeat(simulation.num_envs, 1)
    rotation_tensor = torch.tensor(
        rotation_deg, dtype=torch.float32, device=simulation.device
    ).repeat(simulation.num_envs, 1)
    pose = torch.eye(4, dtype=torch.float32, device=simulation.device).repeat(
        simulation.num_envs, 1, 1
    )
    pose[:, :3, :3] = matrix_from_euler(rotation_tensor * torch.pi / 180.0, "XYZ")
    pose[:, :3, 3] = position_tensor
    return pose


def create_atomic_object(
    simulation: "SimulationManager",
    object_config: Mapping[str, object],
    *,
    initialize: bool = True,
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
                enable_ccd=bool(config.get("enable_ccd", False)),
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
    configured_pose = _configured_pose(simulation, position, rotation)
    if initialize:
        simulation.update(step=int(config.get("settle_steps", 10)))
        entity.clear_dynamics()
        initial_pose = entity.get_local_pose(to_matrix=True).clone()
    else:
        initial_pose = configured_pose
    return AtomicObjectHandle(
        object_id=object_id,
        kind=kind,
        config=config,
        entity=entity,
        initial_pose=initial_pose,
    )


def create_atomic_articulation(
    simulation: "SimulationManager",
    articulation_config: Mapping[str, object],
    *,
    initialize: bool = True,
) -> AtomicArticulationHandle:
    """Create one asset-backed articulation and freeze its settled reset state."""
    object_id = articulation_config.get("id")
    asset_path = articulation_config.get("asset_path")
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("Every atomic articulation must define a non-empty id.")
    if not isinstance(asset_path, str) or not asset_path:
        raise ValueError(
            f"Atomic articulation {object_id!r} must define a non-empty asset_path."
        )
    config = dict(articulation_config)
    position = _vector(
        config.get("position"),
        name=f"articulations[{object_id}].position",
        length=3,
        default=(0.0, 0.0, 0.0),
    )
    rotation = _vector(
        config.get("rotation_deg"),
        name=f"articulations[{object_id}].rotation_deg",
        length=3,
        default=(0.0, 0.0, 0.0),
    )
    scale = _vector(
        config.get("scale"),
        name=f"articulations[{object_id}].scale",
        length=3,
        default=(1.0, 1.0, 1.0),
    )
    raw_qpos = config.get("init_qpos")
    if raw_qpos is not None:
        if not isinstance(raw_qpos, Sequence) or isinstance(raw_qpos, (str, bytes)):
            raise TypeError(
                f"articulations[{object_id}].init_qpos must be a numeric sequence."
            )
        init_qpos = [float(value) for value in raw_qpos]
        if not init_qpos or not all(math.isfinite(value) for value in init_qpos):
            raise ValueError(
                f"articulations[{object_id}].init_qpos must contain finite values."
            )
    else:
        init_qpos = None

    raw_drive = config.get("drive", {})
    raw_attrs = config.get("attrs", {})
    if not isinstance(raw_drive, Mapping):
        raise TypeError(f"articulations[{object_id}].drive must be a mapping.")
    if not isinstance(raw_attrs, Mapping):
        raise TypeError(f"articulations[{object_id}].attrs must be a mapping.")
    resolved_path = Path(asset_path)
    entity = simulation.add_articulation(
        cfg=ArticulationCfg(
            uid=f"atomic_benchmark_{object_id}",
            fpath=str(
                resolved_path
                if resolved_path.is_absolute()
                else get_data_path(asset_path)
            ),
            init_pos=position,
            init_rot=rotation,
            init_qpos=init_qpos,
            drive_pros=JointDrivePropertiesCfg.from_dict(dict(raw_drive)),
            attrs=RigidBodyAttributesCfg.from_dict(dict(raw_attrs)),
            body_scale=scale,
            fix_base=bool(config.get("fix_base", True)),
            disable_self_collision=bool(config.get("disable_self_collision", True)),
            enable_gravity=bool(config.get("enable_gravity", True)),
            min_position_iters=int(config.get("min_position_iters", 4)),
            min_velocity_iters=int(config.get("min_velocity_iters", 1)),
        )
    )
    configured_pose = _configured_pose(simulation, position, rotation)
    configured_qpos = torch.tensor(
        entity.cfg.init_qpos,
        dtype=torch.float32,
        device=simulation.device,
    ).repeat(simulation.num_envs, 1)
    if initialize:
        simulation.update(step=int(config.get("settle_steps", 10)))
        entity.clear_dynamics()
        initial_pose = entity.get_local_pose(to_matrix=True).clone()
        initial_qpos = entity.get_qpos().clone()
    else:
        initial_pose = configured_pose
        initial_qpos = configured_qpos
    return AtomicArticulationHandle(
        object_id=object_id,
        config=config,
        entity=entity,
        initial_pose=initial_pose,
        initial_qpos=initial_qpos,
    )


register_atomic_object_kind("cube", _cube_shape)
register_atomic_object_kind("mesh", _mesh_shape)
