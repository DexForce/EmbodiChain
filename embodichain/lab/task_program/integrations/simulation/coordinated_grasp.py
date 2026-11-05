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

"""Explicit paired grasp services with factory-owned lazy geometry lifetimes."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import MISSING
from typing import Any, TYPE_CHECKING
from embodichain.utils import configclass
from embodichain.toolkits.graspkit.pg_grasp.coordinated import (
    CoordinatedGraspCfg,
    CoordinatedGraspPoseGenerator,
)
from embodichain.lab.sim.utility.grasp_geometry import (
    RobotHandGeometryProvider,
    RobotCoordinatedGraspProvider,
)

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
    from embodichain.toolkits.graspkit import GraspPoseGenerator

__all__ = ["CoordinatedGraspServiceCfg", "ConfiguredCoordinatedGraspServices"]


@configclass
class CoordinatedGraspServiceCfg:
    """Exact paired grasp-target identities and caller-owned contact configuration."""

    left_grasp_target_id: str = MISSING
    right_grasp_target_id: str = MISSING
    contact: CoordinatedGraspCfg = MISSING

    def snapshot(self) -> CoordinatedGraspServiceCfg:
        """Own and validate explicit settings without creating Robot or geometry."""
        self.validate()
        value = deepcopy(self)
        names = (value.left_grasp_target_id, value.right_grasp_target_id)
        if (
            any(
                type(name) is not str or not name or name != name.strip()
                for name in names
            )
            or names[0] == names[1]
        ):
            raise ValueError(
                "Coordinated grasp requires two distinct exact target identities."
            )
        if type(value.contact) is not CoordinatedGraspCfg:
            raise TypeError("Coordinated contact settings require CoordinatedGraspCfg.")
        value.contact.validate_values()
        return value


class _OwnedPairedPort:
    def __init__(self, delegate, validate_owner):
        self._delegate = delegate
        self._validate_owner = validate_owner

    def get_dual_arm_valid_grasp_poses(self, **kwargs):
        self._validate_owner()
        if self._delegate is None:
            raise RuntimeError("Configured paired grasp service is closed.")
        return self._delegate.get_dual_arm_valid_grasp_poses(**kwargs)

    def close(self):
        # Drop owned sampler/geometry caches even if an old engine retains its facade.
        self._delegate = None


class ConfiguredCoordinatedGraspServices:
    """Own paired-query ports for exactly one runtime robot/adapter lifetime.

    Construction retains standalone generators and creates deferred providers;
    geometry capture occurs only on a paired query. Multiple engines/bridges of
    this factory share the same pair cache, while other factories never do.
    """

    def __init__(
        self,
        robot: Robot,
        *,
        generators: Mapping[str, GraspPoseGenerator],
        geometry_cfgs: Mapping[str, RobotHandGeometryCfg],
        declarations: Sequence[CoordinatedGraspServiceCfg],
        validate_owner: Callable[[], None],
    ) -> None:
        """Assemble shared facades without native reads or command changes.

        Args:
            robot: Exact borrowed runtime robot owning both hand calibrations.
            generators: Existing standalone generators, preserved for single calls.
            geometry_cfgs: Explicit hand geometry configurations by grasp target.
            declarations: Unique paired target selections and contact settings.
            validate_owner: Actual factory lifetime guard, invoked before queries.
        """
        if not callable(validate_owner):
            raise TypeError("Coordinated services require a runtime-owner guard.")
        self._ports = []
        self._generators = dict(generators)
        selected = set()
        for raw in declarations:
            if type(raw) is not CoordinatedGraspServiceCfg:
                raise TypeError("Coordinated services require typed declarations.")
            declaration = raw.snapshot()
            left, right = (
                declaration.left_grasp_target_id,
                declaration.right_grasp_target_id,
            )
            if {left, right} & selected:
                raise ValueError(
                    "A grasp target cannot belong to overlapping configured pairs."
                )
            selected.update((left, right))
            if any(
                name not in generators or name not in geometry_cfgs
                for name in (left, right)
            ):
                raise ValueError(
                    "Coordinated service targets require generators and hand geometry."
                )
            if any(geometry_cfgs[name].hand_part != name for name in (left, right)):
                raise ValueError(
                    "Coordinated hand calibration must retain exact grasp target identity."
                )
            provider = RobotCoordinatedGraspProvider(
                generators[left],
                generators[right],
                RobotHandGeometryProvider(robot, cfg=geometry_cfgs[left]),
                RobotHandGeometryProvider(robot, cfg=geometry_cfgs[right]),
                cfg=declaration.contact,
            )
            port = _OwnedPairedPort(provider, validate_owner)
            self._ports.append(port)
            self._generators[left] = CoordinatedGraspPoseGenerator(
                generators[left], port
            )
            self._generators[right] = CoordinatedGraspPoseGenerator(
                generators[right], port
            )

    @property
    def generators(self) -> Mapping[str, GraspPoseGenerator]:
        """Return a fresh mapping of standalone services and pair-enabled facades."""
        return dict(self._generators)

    def close(self) -> None:
        """Invalidate paired ports and release owned geometry/sampler references."""
        for port in self._ports:
            port.close()
        self._ports.clear()
        self._generators.clear()
