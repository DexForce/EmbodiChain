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

"""Owned configuration fragments and deterministic frame composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from ._validation import _identifier, _rigid_pose

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import ArticulationCfg, RigidObjectCfg
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["SceneFragment", "compose", "declare_fragment"]


def _config_pose(cfg: RigidObjectCfg | ArticulationCfg) -> np.ndarray:
    if cfg.init_local_pose is not None:
        return np.asarray(_rigid_pose(cfg.init_local_pose))
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("XYZ", cfg.init_rot, degrees=True).as_matrix()
    pose[:3, 3] = cfg.init_pos
    return np.asarray(_rigid_pose(pose))


class SceneFragment:
    """Own existing entity configurations and named local anchor transforms.

    Args:
        entities: Rigid-object or articulation configurations with unique UIDs.
            Their existing ``init_local_pose`` takes precedence over Euler fields.
        anchors: Optional named 4x4 anchor-to-fragment rigid transforms.

    Configurations are copied on entry and readback. The fragment owns no native
    resources and does not override the deployment's physics configuration.
    """

    def __init__(
        self,
        entities: Sequence[RigidObjectCfg | ArticulationCfg],
        *,
        anchors: Mapping[str, object] | None = None,
    ) -> None:
        from embodichain.lab.sim.cfg import ArticulationCfg, RigidObjectCfg
        from embodichain.lab.sim.cfg import RobotCfg

        owned = deepcopy(tuple(entities))
        seen = set()
        for cfg in owned:
            if not isinstance(cfg, (RigidObjectCfg, ArticulationCfg)) or isinstance(
                cfg, RobotCfg
            ):
                raise TypeError(
                    "Fragments support physical objects; robots belong to embodiments"
                )
            _identifier(cfg.uid, "entity UID")
            if "/" in cfg.uid:
                raise ValueError("Entity UIDs cannot contain a slash (Spawn contract)")
            if cfg.uid in seen:
                raise ValueError(f"Duplicate fragment entity UID: {cfg.uid!r}")
            seen.add(cfg.uid)
            cfg.validate()
            _config_pose(cfg)
        self._entities = owned
        self._anchors = {
            _identifier(name, "anchor name"): _rigid_pose(pose)
            for name, pose in (anchors or {}).items()
        }

    @property
    def entities(self) -> tuple[RigidObjectCfg | ArticulationCfg, ...]:
        """Return owned copies of the entity configurations."""
        return deepcopy(self._entities)

    @property
    def anchors(self) -> dict[str, tuple[tuple[float, ...], ...]]:
        """Return copied anchor names and immutable local transforms."""
        return dict(self._anchors)


def compose(
    fragments: Sequence[SceneFragment],
    *,
    namespace: str = "",
    arena_from_fragment: object | None = None,
    anchor: str | None = None,
) -> SceneFragment:
    """Combine fragments and map an optional anchor into an arena frame.

    Args:
        fragments: Fragments already expressed in the same composition frame.
        namespace: Prefix added to entity and anchor names as ``prefix__name``.
        arena_from_fragment: Desired root transform, or desired anchor transform
            when ``anchor`` is supplied. Defaults to identity.
        anchor: Existing anchor to align with ``arena_from_fragment``.

    Returns:
        New fragment with transformed poses and unchanged body scales. Duplicate
        entity or anchor names are rejected before any simulation mutation.
    """
    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")
    if namespace:
        _identifier(namespace, "namespace")
        if "/" in namespace:
            raise ValueError("namespace cannot contain a slash (Spawn contract)")
    entities = []
    anchors = {}
    for fragment in fragments:
        if not isinstance(fragment, SceneFragment):
            raise TypeError("fragments must contain SceneFragment values")
        entities.extend(fragment.entities)
        for name, pose in fragment.anchors.items():
            if name in anchors:
                raise ValueError(f"Duplicate fragment anchor: {name!r}")
            anchors[name] = pose
    transform = (
        np.eye(4)
        if arena_from_fragment is None
        else np.asarray(_rigid_pose(arena_from_fragment))
    )
    if anchor is not None:
        _identifier(anchor, "anchor")
        if anchor not in anchors:
            raise ValueError(f"Unknown composition anchor: {anchor!r}")
        transform = transform @ np.linalg.inv(np.asarray(anchors[anchor]))
    prefix = f"{namespace}__" if namespace else ""
    for cfg in entities:
        pose = transform @ _config_pose(cfg)
        cfg.uid = prefix + cfg.uid
        cfg.init_local_pose = pose
        cfg.init_pos = tuple(pose[:3, 3])
        cfg.init_rot = tuple(
            Rotation.from_matrix(pose[:3, :3]).as_euler("XYZ", degrees=True)
        )
    return SceneFragment(
        entities,
        anchors={
            prefix + name: transform @ np.asarray(pose)
            for name, pose in anchors.items()
        },
    )


def declare_fragment(
    sim: SimulationManager, fragment: SceneFragment
) -> tuple[str, ...]:
    """Declare a complete fragment through an existing simulation manager.

    Args:
        sim: Caller-owned simulation manager. The caller invokes ``prepare()``.
        fragment: Owned object and articulation configurations to declare.

    Returns:
        Declared UIDs. Existing UIDs are rejected before declaration. Native
        failures propagate and do not promise rollback of earlier declarations.
    """
    from embodichain.lab.sim.cfg import ArticulationCfg

    entities = fragment.entities
    existing = set()
    for kind in (
        "rigid_object",
        "articulation",
        "robot",
        "rigid_object_group",
        "deformable_object",
    ):
        existing.update(getattr(sim, f"get_{kind}_uid_list")())
    duplicate = existing.intersection(cfg.uid for cfg in entities)
    if duplicate:
        raise ValueError(f"Fragment UIDs already exist: {sorted(duplicate)}")
    for cfg in entities:
        method = (
            "add_articulation"
            if isinstance(cfg, ArticulationCfg)
            else "add_rigid_object"
        )
        if not callable(getattr(sim, method, None)):
            raise ValueError(f"Simulation manager does not support {method}")
    for cfg in entities:
        if isinstance(cfg, ArticulationCfg):
            sim.add_articulation(cfg)
        else:
            sim.add_rigid_object(cfg)
    return tuple(cfg.uid for cfg in entities)
