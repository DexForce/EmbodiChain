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

"""Receipt-confirmed task pose coverage shared across sampled scene cases."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .contracts import ExpertEpisode, SceneCase

__all__ = ["TaskPoseCoverage"]


from ._pose_cells import _grid, _pose_cell


class TaskPoseCoverage:
    """Count measured pick/place cells only after ordinary sink confirmation.

    Args:
        domain_id: Host-validated invariant experiment identity, including the
            installation and grid, excluding sampled scene poses and RNG seed.
        entity_uid: Manipulated rigid object whose measured states are counted.
        grids: Explicit ``pick`` and ``place`` XY/yaw grids in arena coordinates.

    Scene cases keep their existing identities. This adapter aggregates them
    within one declared experiment, never across separate instances/domains.
    """

    def __init__(self, domain_id: str, entity_uid: str, grids: Mapping) -> None:
        if not all(
            type(s) is str and s.strip() == s and s for s in (domain_id, entity_uid)
        ):
            raise ValueError("Coverage requires nonempty domain and entity identities")
        if not isinstance(grids, Mapping) or set(grids) != {"pick", "place"}:
            raise ValueError("Coverage requires pick and place grids")
        self.domain_id, self.entity_uid = domain_id, entity_uid
        self.grids = {k: _grid(v) for k, v in grids.items()}
        self._cases: dict[str, tuple[str, str, str]] = {}
        self._population = None
        self._pending: dict[str, tuple[str, tuple]] = {}
        self._confirmed: dict[str, tuple[str, tuple]] = {}
        self._counts = Counter()

    def register_case(
        self,
        case: SceneCase,
        joint_limits: torch.Tensor,
        joint_names: tuple[str, ...],
        reference: float | None,
    ) -> None:
        """Register a scene without changing its identity or normalization."""
        if reference is not None:
            raise ValueError(
                "Task pose coverage does not accept a manipulability reference"
            )
        population = (
            case.task_id,
            case.robot_profile_id,
            case.calibration_id,
            tuple(joint_names),
        )
        if self._population is not None and population != self._population:
            raise ValueError(
                "Coverage domain cannot mix task/robot/calibration/joint layouts"
            )
        self._population = population
        identity = (case.task_id, case.robot_profile_id, case.calibration_id)
        if (
            case.scene_case_id in self._cases
            and self._cases[case.scene_case_id] != identity
        ):
            raise ValueError("Conflicting scene case registration")
        self._cases[case.scene_case_id] = identity

    def reference(self, case: SceneCase) -> None:
        """This coverage policy supplies no motion normalization reference."""
        if case.scene_case_id not in self._cases:
            raise KeyError(case.scene_case_id)
        return None

    def cells(
        self, pick_pose: object, place_pose: object
    ) -> tuple[tuple[str, int, int, int], ...]:
        """Return ordered tagged cells; out-of-domain poses contribute no cell."""
        return tuple(
            (phase, *cell)
            for phase, pose in (("pick", pick_pose), ("place", place_pose))
            if (cell := _pose_cell(pose, self.grids[phase])) is not None
        )

    def reserve(self, episode: ExpertEpisode) -> bool:
        """Own measured cells until the matching persistence receipt arrives."""
        case_id = episode.identity.scene_case_id
        if case_id not in self._cases:
            raise ValueError("Episode scene is outside this coverage domain")
        if not episode.validation.accepted:
            return False
        states = episode.observations.get(f"rigid/{self.entity_uid}/state")
        if (
            states is None
            or states.ndim != 2
            or states.shape[0] != len(episode.timestamps)
            or states.shape[1] < 7
        ):
            raise ValueError("Task coverage requires T+1 measured rigid states")
        cells = self.cells(
            states[0].detach().cpu().numpy(), states[-1].detach().cpu().numpy()
        )
        value = (case_id, cells)
        previous = self._confirmed.get(
            episode.commit_id, self._pending.get(episode.commit_id)
        )
        if previous is not None and previous != value:
            raise ValueError("Commit identity was reused with different measured cells")
        if episode.commit_id not in self._confirmed:
            self._pending[episode.commit_id] = value
        return True  # Duplicates are valid data; they cannot increase unique coverage.

    def confirm(self, case_id: str, commit_id: str) -> None:
        """Idempotently count the reservation qualified by the session's receipt."""
        value = self._confirmed.get(commit_id, self._pending.get(commit_id))
        if value is None or value[0] != case_id:
            raise ValueError("Foreign or unreserved coverage confirmation")
        if commit_id in self._confirmed:
            return
        self._pending.pop(commit_id)
        self._confirmed[commit_id] = value
        self._counts.update(value[1])

    def release(self, case_id: str, commit_id: str) -> None:
        """Release unconfirmed evidence without erasing committed coverage."""
        value = self._pending.get(commit_id)
        if value is not None and value[0] != case_id:
            raise ValueError("Foreign coverage release")
        self._pending.pop(commit_id, None)

    def gain(self, cells: tuple, predicted: Mapping | None = None) -> float:
        """Score scarcity using formal counts and explicitly tentative occupancy."""
        pending = Counter(
            cell for _, values in self._pending.values() for cell in values
        )
        predicted = {} if predicted is None else predicted
        return sum(
            1 / (1 + self._counts[cell] + pending[cell] + predicted.get(cell, 0))
            for cell in cells
        )

    def snapshot(self) -> tuple[Mapping[str, int], Mapping[str, Mapping[int, int]]]:
        """Return confirmed coverage with no fabricated manipulability bands."""
        return {self.domain_id: len(self._counts)}, {}

    def phase_counts(self) -> Mapping[str, int]:
        """Return owned pick/place counts separately from motion quality bands."""
        return {
            phase: sum(cell[0] == phase for cell in self._counts)
            for phase in self.grids
        }

    def state_dict(self) -> dict:
        """Return an owned receipt-derived checkpoint, including pending evidence."""
        return deepcopy(
            {
                "version": 1,
                "domain_id": self.domain_id,
                "entity_uid": self.entity_uid,
                "grids": self.grids,
                "population": self._population,
                "cases": self._cases,
                "pending": self._pending,
                "confirmed": self._confirmed,
            }
        )

    def load_state_dict(self, state: Mapping) -> None:
        """Restore matching-domain counts; pending evidence remains unconfirmed."""
        data = deepcopy(dict(state))
        if data.get("version") != 1 or any(
            data.get(k) != getattr(self, k)
            for k in ("domain_id", "entity_uid", "grids")
        ):
            raise ValueError("Coverage checkpoint belongs to another domain/grid")
        if self._pending or self._confirmed or self._cases:
            raise ValueError("Restore coverage only into a new adapter")

        def decode(entries):
            return {
                key: (value[0], tuple(tuple(cell) for cell in value[1]))
                for key, value in entries.items()
            }

        pending, confirmed = decode(data["pending"]), decode(data["confirmed"])
        if pending.keys() & confirmed.keys():
            raise ValueError("Coverage checkpoint repeats pending/confirmed identities")
        cases = {k: tuple(v) for k, v in data["cases"].items()}
        for case, cells in (*pending.values(), *confirmed.values()):
            if case not in cases or any(
                len(c) != 4
                or c[0] not in self.grids
                or any(type(i) is not int or i < 0 for i in c[1:])
                or any(
                    i >= round((hi - lo) / step)
                    for i, lo, hi, step in zip(
                        c[1:],
                        self.grids[c[0]]["lower"],
                        self.grids[c[0]]["upper"],
                        self.grids[c[0]]["cell_size"],
                    )
                )
                for c in cells
            ):
                raise ValueError("Malformed coverage checkpoint cells")
        pop = data["population"]
        self._cases = cases
        self._population = None if pop is None else (*pop[:3], tuple(pop[3]))
        self._pending, self._confirmed = pending, confirmed
        self._counts = Counter(
            cell for _, cells in confirmed.values() for cell in cells
        )
