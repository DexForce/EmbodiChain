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

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, TypeVar, Sequence

from embodichain.lab.sim.cfg import ObjectBaseCfg
from embodichain.utils import logger
import dexsim
from copy import deepcopy

T = TypeVar("T")


@dataclass
class BatchEntity(ABC):
    """Abstract base class for batch entity in the simulation engine.

    This class defines the interfaces for managing and manipulating a batch of entity.
    A single entity could be one of the following assets:
    - actor (eg. rigid object)
    - articulation (eg. robot)
    - camera
    - light
    - sensor (eg. force sensor)

    """

    uid: str | None = None
    cfg: ObjectBaseCfg = None
    _entities: List[T] = None
    device: torch.device = None

    def __init__(
        self,
        cfg: ObjectBaseCfg,
        entities: List[T] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:

        if entities is None or len(entities) == 0:
            logger.log_error("Invalid entities list: must not be empty.")

        self.cfg = deepcopy(cfg)
        self.uid = self.cfg.uid
        if self.uid is None:
            logger.log_error("UID must be set in the configuration.")
        self._entities = entities
        self.device = device

    def __str__(self) -> str:
        return f"{self.__class__}: managing {self.num_instances} {self._entities[0].__class__} objects | uid: {self.uid} | device: {self.device}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def num_instances(self) -> int:
        return len(self._entities)

    @abstractmethod
    def set_local_pose(
        self, pose: torch.Tensor, env_ids: Sequence[int] | None = None
    ) -> None:
        pass

    @abstractmethod
    def get_local_pose(self, to_matrix: bool = False) -> torch.Tensor:
        pass

    @property
    def pose(self) -> torch.Tensor:
        return self.get_local_pose(to_matrix=False)

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the entity to its initial state.

        Args:
            env_ids (Sequence[int] | None): The environment IDs to reset. If None, reset all environments.
        """
        pass

    def get_visual_elements(self) -> tuple[dict[str, object], ...]:
        """Return owned metadata snapshots for configured or USD source visuals.

        Gaussian arrays are not copied. Resource IDs are diagnostics scoped to
        the current Scene; USD-native visual references do not expose them.
        Configuration element IDs and USD prim paths remain stable.

        Returns:
            Element metadata with representation, local transform and resource IDs.
        """
        binding = getattr(self, "_visual_binding", None)
        if binding is not None:
            return binding.describe()
        source_elements = getattr(self, "_source_visual_elements", None)
        if source_elements is not None:
            from copy import deepcopy

            return deepcopy(source_elements)
        if getattr(self.cfg, "source", None) is not None and getattr(
            self, "is_spawn_bound", False
        ):
            from embodichain.lab.sim.spawn.usd import _source_visual_elements

            entities = getattr(self, "_entities", ())
            descriptor = getattr(entities[0], "object_desc", None) if entities else None
            if descriptor is not None:
                return _source_visual_elements(descriptor)
        from embodichain.lab.sim.cfg.visuals import GaussianSplatCfg, _visual_elements

        groups = getattr(self.cfg, "link_visuals", {})
        if getattr(self.cfg, "visuals", None) is not None:
            groups = {None: self.cfg.visuals}
        return tuple(
            {
                "id": element.id,
                "link_name": link,
                "representation": (
                    "gaussian_splat"
                    if isinstance(element.representation, GaussianSplatCfg)
                    else "mesh"
                ),
                "file_path": getattr(
                    element.representation,
                    "file_path",
                    getattr(element.representation, "fpath", None),
                ),
                "local_pose": element.local_pose.copy(),
                "scale": element.scale,
                "resource_ids": (),
            }
            for link, elements in groups.items()
            for element in (_visual_elements(elements) or ())
        )

    def set_visual_visibility(
        self, visible: bool, env_ids: Sequence[int] | None = None
    ) -> None:
        """Set visibility of configured visual elements in selected environments.

        Args:
            visible: Whether the configured elements should be rendered.
            env_ids: Environment indices, or None for all environments.

        Raises:
            ValueError: If the entity has no explicit visual composition.
        """
        binding = getattr(self, "_visual_binding", None)
        if binding is None:
            from .cfg.visuals import _uses_descriptor_visuals

            if _uses_descriptor_visuals(self.cfg):
                self.set_visible(visible, env_ids=env_ids)
                return
            raise ValueError("Explicit visuals are required for this operation.")
        binding.set_visible(visible, env_ids)

    def set_gaussian_visual_state(
        self,
        element_id: str,
        env_ids: Sequence[int] | None = None,
        *,
        link_name: str | None = None,
        color_gain: object = None,
        opacity_gain: object = None,
        scale: object = None,
        visible: object = None,
        lod_enabled: object = None,
        lod_detail_scale: object = None,
    ) -> None:
        """Stage Gaussian instance randomization for the next managed render.

        The shared immutable point data and physical properties remain unchanged.
        Reset restores the configured baseline for selected environments. Declare
        several Gaussian elements and select their visibility for an asset bank;
        each source is shared across arenas and charged once to residency.

        Args:
            element_id: Stable visual element ID within its owner/link.
            env_ids: Selected environments, or None for all.
            link_name: Articulation link name, or None for a root/body visual.
            color_gain: RGB radiance multipliers, shape (3,) or (N,3).
            opacity_gain: Nonnegative opacity multiplier, scalar or (N,).
            scale: Positive uniform geometry scale including owner body scale,
                scalar or (N,). Does not scale physics or the local translation.
            visible: Element visibility, boolean scalar or (N,).
            lod_enabled: Hierarchy selection, boolean scalar or (N,).
            lod_detail_scale: Positive detail multiplier, scalar or (N,).

        Raises:
            ValueError: No matching Gaussian element or invalid state values.
        """
        binding = getattr(self, "_visual_binding", None)
        if binding is None:
            raise ValueError(
                "Explicit Gaussian visuals are required for this operation."
            )
        binding.set_gaussian_state(
            element_id,
            env_ids,
            link_name=link_name,
            color_gain=color_gain,
            opacity_gain=opacity_gain,
            geometry_scales=scale,
            visible=visible,
            lod_enabled=lod_enabled,
            lod_detail_scale=lod_detail_scale,
        )

    def destroy(self) -> None:
        """Destroy all entities managed by this batch entity."""
        pass
