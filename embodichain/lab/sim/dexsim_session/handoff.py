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

"""EmbodiChain provider for an in-place handoff on a standard DexSim host."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import os
from uuid import uuid4
from typing import Any, TYPE_CHECKING

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.robot_registry import resolve_robot_cfg

from .control import WorldControl
from .writer import writer_executors
from . import robot_state

if TYPE_CHECKING:
    from dexsim.runtime import SessionContext

__all__: list[str] = []


class BoundWorld:
    """Resolve explicit roles against live paths and construct task-local views."""

    read_modules = (robot_state.__name__,)

    @staticmethod
    def read(context: SessionContext, params: dict[str, Any]) -> dict[str, Any]:
        return robot_state.run(context, **params["arguments"])

    def __init__(self, context: SessionContext, key: bytes, binding: dict[str, Any]):
        self.resources = ExitStack()
        self.motion = None
        if binding.get("profile") in ("anymal-c", "rp1"):
            from ._motion_jobs import executors
            from .live_policy import create_policy, operations

            output = Path(os.environ["EMBODICHAIN_MOTION_OUTPUT"]).resolve()
            output.mkdir(parents=True, exist_ok=True)
            self.motion = create_policy(context, binding, output)
            self.motion.jobs.binding_id = uuid4().hex
            self.resources.callback(self.motion.close)
            self.invoke = executors(context, operations(self.motion), key)[
                "adapter.invoke"
            ]
            return
        if binding.get("profile") == "ur5-pick-place":
            from .ur5.ur5_pick_place import _Episode
            from .ur5._ur5_motion import Ur5Motion
            from ._motion_jobs import executors

            # Deployment/assets/output are selected by the operator, not by a tool caller.
            deployment = Path(os.environ["EMBODICHAIN_UR5_DEPLOYMENT"]).resolve(
                strict=True
            )
            output = Path(os.environ["EMBODICHAIN_MOTION_OUTPUT"]).resolve()
            output.mkdir(mode=0o700, parents=True, exist_ok=True)
            with ExitStack() as constructing:
                episode = _Episode(
                    context,
                    deployment,
                    False,
                    constructing,
                    live_paths=binding["paths"],
                    destinations=binding["destinations"],
                )
                self.motion = Ur5Motion(episode, output)
                self.motion.jobs.binding_id = uuid4().hex
                constructing.callback(self.motion.close)
                self.invoke = executors(context, self.motion.operations(), key)[
                    "adapter.invoke"
                ]
                self.resources = constructing.pop_all()
            return
        robots = []
        objects = []
        paths = {}
        for uid, spec in binding["robots"].items():
            cfg = resolve_robot_cfg(spec["robot_cfg"], variant=spec.get("variant"))
            cfg.uid = uid
            robots.append(cfg)
            paths[uid] = spec["object_ref"]
        for uid, spec in binding["bodies"].items():
            objects.append(RigidObjectCfg(uid=uid, body_type=spec["body_type"]))
            paths[uid] = spec["object_ref"]
        self.manager = SimulationManager.attach(
            context,
            arena=binding["arena"],
            robots=robots,
            rigid_objects=objects,
            paths=paths,
            logical_entities=binding.get("roles"),
        )
        try:
            self.control = WorldControl(
                context,
                self.manager,
                physics_dt=binding["physics_dt"],
                steps_per_control=binding["steps_per_control"],
            )
            self.invoke = writer_executors(self.control, key)["adapter.invoke"]
        except BaseException:
            self.manager.detach()
            raise

    def revoke(self) -> None:
        """Interrupt an active provider job at the host callback boundary."""
        if self.motion is not None:
            self.motion.jobs.close()
        else:
            self.control.stop()

    def close(self) -> None:
        if self.motion is not None:
            if self.motion.jobs.active is not None:
                raise RuntimeError(
                    "Stop the active job at a control boundary before releasing"
                )
            self.resources.close()
            return
        if self.control.stop()["state"] == "effect_unknown":
            raise RuntimeError("Cannot release a World with unknown effects")
        self.manager.detach()
