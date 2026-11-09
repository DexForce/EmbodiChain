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

"""Capture exact typed Default collision parts for B=1 free-motion queries."""

from __future__ import annotations

from typing import TYPE_CHECKING
import math

from .collision_sources import CollisionObjectSnapshot
from .feasibility.collision import UnsupportedFeasibilityError

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__: list[str] = []


def _capture_free_motion_sources(
    simulation: SimulationManager, *, include_rigids: bool
) -> dict[str, CollisionObjectSnapshot]:
    """Own all selected rigid parts without changing shape semantics or scale.

    Articulated world capture is outside this E1 service. Its presence rejects
    the complete query instead of silently omitting any required geometry.
    """
    if simulation.num_envs != 1:
        raise ValueError("Free-motion collision capture requires B=1.")
    if not simulation.is_default_backend:
        raise UnsupportedFeasibilityError(
            "Free-motion capture requires Default physics."
        )
    if simulation.get_articulation_uid_list():
        raise UnsupportedFeasibilityError(
            "Free-motion articulation-world capture is not available in this capability."
        )
    result: dict[str, CollisionObjectSnapshot] = {}
    scene = simulation.spawn_result
    handles = [scene.get(path) for path in scene.get_paths()]
    for uid in simulation.get_rigid_object_uid_list() if include_rigids else ():
        obj = simulation.get_rigid_object(uid)
        selected = [handle for handle in handles if handle.get_name() == uid]
        if len(selected) != 1:
            raise UnsupportedFeasibilityError(
                f"Native rigid ownership is unavailable or ambiguous: {uid!r}."
            )
        body = selected[0].native().get_physical_body()
        if body is None or not body.is_collision_enable():
            raise UnsupportedFeasibilityError(
                f"Required native rigid collision is unavailable or disabled: {uid!r}."
            )
        if not all(
            math.isfinite(value)
            for value in (
                float(body.get_rest_offset()),
                float(body.get_contact_offset()),
            )
        ):
            raise UnsupportedFeasibilityError("Native rigid offsets are unavailable.")
        shapes = obj.get_collision_shapes(env_id=0)
        if not shapes or len(shapes) != body.get_shape_count():
            raise UnsupportedFeasibilityError(
                f"Required collision shape inventory is empty or incomplete: {uid!r}."
            )
        pose = obj.get_local_pose(to_matrix=True)[0:1].detach().cpu()
        for index, shape in enumerate(shapes):
            result[f"{uid}:{index}"] = CollisionObjectSnapshot(
                (shape,), pose, provenance="rigid_physical_collision_descriptors"
            )
    return result
