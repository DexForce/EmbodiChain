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

"""Isaac-style flat marker updates backed by the shared marker group."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import numpy as np

from embodichain.utils import configclass
from .._utils import to_numpy_array
from .cfg import MarkerGroupCfg, MarkerPrototypeCfg

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__ = ["VisualizationMarkers", "VisualizationMarkersCfg"]


@configclass
class VisualizationMarkersCfg:
    """Flat marker configuration using xyzw quaternions.

    ``prim_path`` is a unique registry name, not a USD scene path. Dictionary
    order defines marker indices. World coordinates are the default; ``env``
    adds the selected environment's origin once. No environment layout is
    inferred from the number of instances.
    """

    prim_path: str = MISSING
    markers: dict[str, MarkerPrototypeCfg] = MISSING
    coordinate_space: Literal["world", "env"] = "world"


class VisualizationMarkers:
    """Adapt flat Isaac-style arrays to a manager-owned MarkerGroup.

    Args:
        cfg: Prototypes and coordinate convention.
        sim: Simulation owning the marker resources. Native rendering requires
            an explicitly prepared scene.
    """

    def __init__(self, cfg: VisualizationMarkersCfg, sim: SimulationManager) -> None:
        cfg.validate()
        if cfg.coordinate_space not in {"world", "env"}:
            raise ValueError("coordinate_space must be 'world' or 'env'.")
        self._coordinate_space = cfg.coordinate_space
        self._group = sim.add_marker_group(
            MarkerGroupCfg(name=cfg.prim_path, prototypes=cfg.markers)
        )
        self._environment_ids = np.empty(0, dtype=np.int64)
        self._visible = True

    @property
    def count(self) -> int:
        """Number of logical instances, including hidden instances."""
        return self._group.count

    @property
    def num_prototypes(self) -> int:
        """Number of configured marker prototypes."""
        return len(self._group._cfg.prototypes)

    def is_visible(self) -> bool:
        """Return the visibility requested through this facade."""
        return self._visible

    def set_visibility(self, visible: bool) -> None:
        """Set visibility while retaining instance state.

        Args:
            visible: Whether the group is displayed.
        """
        self._group.set_visibility(visible)
        self._visible = bool(visible)

    def visualize(
        self,
        translations: object | None = None,
        orientations: object | None = None,
        scales: object | None = None,
        marker_indices: object | None = None,
        environment_ids: object | None = None,
    ) -> None:
        """Update a complete flat instance selection atomically.

        Omitted arrays retain previous values when the count is unchanged.
        Translations determine a new count and reset omitted fields on resize.
        Instances may be interleaved across environments. Omitted environment
        IDs retain membership, or select environment zero after a resize.
        Hidden groups accept updates and show the latest state when enabled.

        Args:
            translations: Finite positions shaped (N, 3).
            orientations: Nonzero xyzw quaternions shaped (N, 4).
            scales: Positive scales shaped (N, 3).
            marker_indices: Prototype indices shaped (N,).
            environment_ids: Integer environment IDs shaped (N,). Changing
                membership requires translations in the same call.
        """
        self._group._check_live()
        provided = {
            key: to_numpy_array(value, np.float64)
            for key, value in {
                "translations": translations,
                "orientations_xyzw": orientations,
                "scales": scales,
                "prototype_indices": marker_indices,
            }.items()
            if value is not None
        }
        if not provided and environment_ids is None:
            return
        count = self.count
        if "translations" in provided:
            if provided["translations"].ndim != 2:
                raise ValueError("translations must have shape (N, 3).")
            count = len(provided["translations"])
        resized = count != self.count
        if environment_ids is None:
            ids = (
                np.zeros(count, dtype=np.int64)
                if resized
                else self._environment_ids.copy()
            )
        else:
            raw = to_numpy_array(environment_ids, None)
            if raw.shape != (count,) or (count and raw.dtype.kind not in "iu"):
                raise ValueError(
                    "environment_ids must be an integer array shaped (N,)."
                )
            if np.any(raw < 0) or np.any(raw >= self._group.num_envs):
                raise ValueError("environment_ids are out of bounds.")
            ids = raw.astype(np.int64)
            if translations is None and not np.array_equal(ids, self._environment_ids):
                raise ValueError("Changing environment_ids requires translations.")
        arrays = self._group._defaults(count)
        if not resized and count:
            # Restore caller ordering from the authoritative per-environment state.
            for env_id, state in enumerate(self._group._states):
                rows = self._environment_ids == env_id
                for key in arrays:
                    if state[key] is not None:
                        arrays[key][rows] = state[key]
        for key, array in provided.items():
            width = {"translations": 3, "orientations_xyzw": 4, "scales": 3}.get(key)
            expected = (count, width) if width else (count,)
            if array.shape != expected:
                raise ValueError(f"{key} must have shape {expected}.")
            arrays[key] = array
        if translations is not None and self._coordinate_space == "world":
            arrays["translations"] = arrays["translations"] - self._group._origins[ids]
        states = [
            self._group._updated_state(
                env_id,
                {
                    key: array[ids == env_id]
                    for key, array in arrays.items()
                    if array is not None
                },
            )
            for env_id in range(self._group.num_envs)
        ]
        self._group._commit(on_change=self._group._on_change, states=states)
        self._environment_ids = ids

    def remove(self) -> None:
        """Release all marker resources; repeated removal is safe."""
        self._group.remove()
