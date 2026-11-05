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

"""Atomic-owned source-compatible joint-prefix calls to observed cuRobo queries."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, ContextManager
import torch

from embodichain.lab.sim.motion.motion_generator import (
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners.curobo.curobo_planner import CuroboPlanOptions
from embodichain.lab.sim.motion.planners.curobo.observed import (
    ObservedCuroboQueryFactory,
    CuroboWorldObservation,
)
from embodichain.lab.sim.motion.planners.utils import PlanResult, PlanState
from embodichain.lab.sim.motion.planners.curobo.payload import CuroboPayload
from .planning_probes import CuroboWorldObservationProvider
from .state import PlanningContext

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["CuroboJointPrefixPlanner"]


class CuroboJointPrefixPlanner:
    """Preserve explicit joint goals and the source Torch-only planning boundary.

    Args:
        queries: Shared Motion factory owning fresh observed static worlds.
        world_provider: Host adapter supplying the complete same-observation
            world roster, including collision-only links and actual native stamp.

    The adapter adds no stage selection or recovery iteration. It uses absent
    velocity/acceleration overrides and preserves native samples, matching the
    source direct-prefix route. Payload transport retains its own explicit
    policy choices and uses the underlying query factory separately.
    """

    def __init__(
        self,
        queries: ObservedCuroboQueryFactory,
        world_provider: CuroboWorldObservationProvider,
    ) -> None:
        if not isinstance(world_provider, CuroboWorldObservationProvider):
            raise TypeError(
                "A joint prefix requires a typed world observation provider."
            )
        self._queries, self._world_provider = queries, world_provider

    @property
    def robot(self) -> Robot:
        """Return the exact borrowed robot planned by this adapter."""
        return self._queries.robot

    def planning_device(self) -> torch.device:
        """Resolve the shared factory's selected backend device."""
        return self._queries.planning_device

    def open(
        self,
        *,
        context: PlanningContext,
        control_part: str,
        full_qpos: torch.Tensor,
        payload: CuroboPayload | None = None,
        world_metadata: dict[str, object] | None = None,
    ) -> ContextManager[MotionGenerator]:
        """Prepare an observation-bound query without selecting a motion or RNG law.

        Args:
            context: Captured B1/env0 robot root and scene scope.
            control_part: Explicit active group.
            full_qpos: Complete locked planning state.
            world_metadata: Optional caller-owned diagnostics receiving the exact world roster.
            payload: Optional explicit carried-body envelope.

        Returns:
            A lazy backend lifetime yielding the standard motion facade. The
            world observation is prepared now, before the caller enters its
            chosen planning RNG boundary. Acquisition is delegated to the
            supplied provider; no unscoped root/world read is substituted.
        """
        if context.batch_size != 1 or context.env_ids.tolist() != [0]:
            raise ValueError("Observed cuRobo planning supports only B1/env0.")
        if context.robot.root_pose is None:
            raise ValueError(
                "Observed cuRobo planning requires a captured robot root pose."
            )
        if context.scene.timestamp != context.robot.timestamp:
            raise ValueError("Observed robot and scene times disagree.")
        if control_part not in self.robot.control_parts:
            raise ValueError("Observed query control part is unavailable.")
        if (
            not isinstance(full_qpos, torch.Tensor)
            or full_qpos.shape != context.robot.qpos.shape
        ):
            raise ValueError("Observed query requires the complete locked joint scope.")
        if payload is not None and (
            type(payload) is not CuroboPayload or payload.control_part != control_part
        ):
            raise ValueError("Observed query payload does not match its control part.")
        dt = context.require_control_dt()
        world = (
            self._world_provider.capture(context)
            if payload is None
            else self._world_provider.capture(
                context, excluded_entity_ids=(payload.object_id,)
            )
        )
        if type(world) is not CuroboWorldObservation:
            raise TypeError("World provider returned an untyped observation.")
        if (
            world.scene_version != context.scene.version
            or world.timestamp != context.robot.timestamp
        ):
            raise ValueError("World is from a different planning observation.")
        return self._queries.open(
            control_part=control_part,
            full_qpos=full_qpos,
            root_pose=context.robot.root_pose,
            root_link_name=self.robot.root_link_name,
            observed_at=context.robot.timestamp,
            world=world,
            control_dt=dt,
            payload=payload,
            **(
                {"world_metadata": world_metadata} if world_metadata is not None else {}
            ),
        )

    def plan(
        self,
        *,
        context: PlanningContext,
        control_part: str,
        start_qpos: torch.Tensor,
        target_qpos: torch.Tensor,
        full_qpos: torch.Tensor,
        control_dt: float,
    ) -> PlanResult:
        """Plan the exact controlled-joint endpoint without changing caller state.

        Args:
            context: Current B1/env0 root, joints and scene observation.
            control_part: Exact active group owned by the selected robot.
            start_qpos: Controlled-joint start in this query's full joint input.
            target_qpos: Exact controlled-joint target, never resampled here.
            full_qpos: Explicit full locked planning state in native joint order.
            control_dt: Caller-owned command interval.

        Returns:
            Owned native planner samples after backend cleanup. Endpoint/grid
            composition and final path qualification stay with the Atomic caller.
        """
        if context.batch_size != 1 or context.env_ids.tolist() != [0]:
            raise ValueError("Source joint-prefix planning supports only B1/env0.")
        if context.robot.root_pose is None:
            raise ValueError(
                "Joint-prefix planning requires a captured robot root pose."
            )
        if context.scene.timestamp != context.robot.timestamp:
            raise ValueError("Joint-prefix robot and scene observation times disagree.")
        if control_dt != context.require_control_dt():
            raise ValueError("Joint-prefix cadence differs from the planning context.")
        if control_part not in self.robot.control_parts:
            raise ValueError("Joint-prefix control part is unavailable.")
        ids = [
            self.robot.joint_names.index(name)
            for name in self.robot.control_parts[control_part]
        ]
        if (
            not isinstance(full_qpos, torch.Tensor)
            or full_qpos.shape != context.robot.qpos.shape
            or start_qpos.shape != (1, len(ids))
            or target_qpos.shape != start_qpos.shape
            or not torch.equal(full_qpos[:, ids], start_qpos)
            or not torch.isfinite(target_qpos).all()
        ):
            raise ValueError(
                "Joint-prefix target/start must match the exact full-state scope."
            )
        query = self.open(
            context=context, control_part=control_part, full_qpos=full_qpos
        )
        device = self.planning_device()
        devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        device_scope = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        # Preserve the original CPU/current-planning-CUDA RNG scope. Python and
        # NumPy random streams are deliberately not reset by this adapter.
        with device_scope, torch.random.fork_rng(devices=devices):
            with query as generator:
                return generator.generate(
                    [PlanState.from_qpos(target_qpos)],
                    options=MotionGenOptions(
                        strategy="motion_gen",
                        start_qpos=start_qpos,
                        control_part=control_part,
                        sample_count=None,
                        velocity_limit=None,
                        acceleration_limit=None,
                        plan_opts=CuroboPlanOptions(),
                    ),
                )
