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

"""Configuration of bounded, explicit scene proposals."""

from __future__ import annotations

from embodichain.utils import configclass

from .contracts import _identifier, _nonnegative_integer

__all__ = ["SceneExpansionCfg"]


@configclass
class SceneExpansionCfg:
    """Limits and permissions for a scene candidate source.

    Args:
        seed: Nonnegative proposal-generation seed; not a physical replay seed.
        max_candidates: Positive maximum candidates consumed from the source.
        movable_entity_ids: Unique physical UIDs allowed to change pose. An
            empty tuple permits only nominal candidates with no pose changes.

    Execution and persistence budgets remain owned by the host integration.
    Calling a scene expansion integration opts in; ordinary execution paths
    do not consult this configuration.
    """

    seed: int = 0
    max_candidates: int = 1
    movable_entity_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonnegative_integer(self.seed, "seed")
        _nonnegative_integer(self.max_candidates, "max_candidates")
        if self.max_candidates == 0:
            raise ValueError("max_candidates must be positive")
        if isinstance(self.movable_entity_ids, str):
            raise ValueError("movable_entity_ids must be a sequence of entity UIDs")
        try:
            self.movable_entity_ids = tuple(self.movable_entity_ids)
        except TypeError as exc:
            raise ValueError(
                "movable_entity_ids must be a sequence of entity UIDs"
            ) from exc
        for uid in self.movable_entity_ids:
            _identifier(uid, "movable_entity_ids entry")
        if len(set(self.movable_entity_ids)) != len(self.movable_entity_ids):
            raise ValueError("movable_entity_ids must contain unique entity UIDs")
