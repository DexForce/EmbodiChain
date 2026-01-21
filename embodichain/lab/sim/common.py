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

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, TypeVar, Sequence

from embodichain.lab.sim.cfg import ObjectBaseCfg
from embodichain.utils import logger
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

        self.reset()

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

    def destroy(self) -> None:
        """Destroy all entities managed by this batch entity."""
        pass
