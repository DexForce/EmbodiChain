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

"""Resident marker resources and one cross-group native frame publication."""

from __future__ import annotations

from typing import TYPE_CHECKING
from .group import MarkerGroup
from .cfg import MarkerGroupCfg
from ._native import NativeMarkerRenderer

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__ = ["MarkerFramePublisher"]


class MarkerFramePublisher:
    """Stage robot/object groups and publish one coherent render-only frame.

    Args:
        simulation: Prepared simulation owning the native render resources.
        xray: Disable depth culling for translucent comparison overlays.

    Groups belong to this publisher, independently of automatic simulation
    markers. Initial publication uploads all declared resources, including hidden
    instances. Later publications transform only visible instances, retain hidden
    resources, and use actual native visibility rather than moving/scaling objects.
    Publishing neither prepares nor steps the simulation. Close before the World.
    """

    def __init__(self, simulation: SimulationManager, *, xray: bool = False) -> None:
        self.sim = simulation
        self.groups: dict[str, MarkerGroup] = {}
        self.native = NativeMarkerRenderer(simulation.spawn_result, xray=xray)
        self.publications = 0
        self._closed = False

    def add_marker_group(self, cfg: MarkerGroupCfg) -> MarkerGroup:
        """Declare a group compatible with RobotMarkers/VisualizationMarkers."""
        if self._closed:
            raise RuntimeError("marker frame publisher is closed")
        if cfg.name in self.groups:
            raise ValueError(f"duplicate marker group: {cfg.name}")
        group = MarkerGroup(
            cfg, num_envs=self.sim.num_envs, origins=self.sim.arena_offsets
        )
        self.groups[group.name] = group
        return group

    def publish(self, *, preload: bool = False) -> None:
        """Submit all staged groups together, retaining hidden GPU resources."""
        if self._closed:
            raise RuntimeError("marker frame publisher is closed")
        initial = self.publications == 0 or preload
        meshes = tuple(
            mesh
            for group in self.groups.values()
            for mesh in group.snapshot(visible_only=not initial)
        )
        self.native.publish("frame", meshes, retain_omitted=True)
        self.publications += 1

    def close(self) -> None:
        """Release resident batches, meshes and groups; repeated close is safe."""
        if self._closed:
            return
        self._closed = True
        self.native.close()
        for group in self.groups.values():
            group.remove()
        self.groups.clear()
