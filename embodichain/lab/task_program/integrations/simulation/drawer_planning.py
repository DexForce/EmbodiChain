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

"""Declared drawer occurrences assembled over live Scene and Motion owners."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
import numpy as np
import torch
from embodichain.lab.sim.scene.articulation_inputs import PrismaticBinding
from embodichain.lab.sim.scene.drawer_observation import DrawerPayloadObservation
from embodichain.lab.sim.atomic_actions.drawer_planning import DrawerPlanningInputs
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.sim.motion.mesh_collision import RobotArticulationCollisionSnapshot

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.atomic_actions import (
        ResolvedActionRequest,
        PlanningContext,
    )

__all__ = ["DrawerInvocationBinding", "SimulationDrawerPlanningProvider"]


@dataclass(frozen=True, slots=True)
class DrawerInvocationBinding:
    """Bind a complete execution occurrence to one physical drawer payload."""

    invocation_id: str
    operation: Literal["place", "transport", "close"]
    target_id: str
    link_entity_id: str
    object_id: str
    source: PrismaticBinding
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]

    def __post_init__(self) -> None:
        for name in ("invocation_id", "target_id", "link_entity_id", "object_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or value.strip() != value:
                raise ValueError(f"Drawer {name} must be an exact identity.")
        if self.operation not in ("place", "transport", "close"):
            raise ValueError("Unknown drawer operation.")
        if type(self.source) is not PrismaticBinding:
            raise TypeError("Drawer source must be a qualified PrismaticBinding.")
        bounds = np.asarray([self.lower, self.upper], dtype=float)
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or (bounds[0] > bounds[1]).any()
        ):
            raise ValueError("Drawer bounds must be finite ordered 3D values.")
        object.__setattr__(self, "lower", tuple(map(float, self.lower)))
        object.__setattr__(self, "upper", tuple(map(float, self.upper)))


class SimulationDrawerPlanningProvider:
    """Resolve explicit occurrences; collect facts without selecting skill behavior."""

    def __init__(
        self,
        robot: Robot,
        simulation: SimulationManager,
        *,
        bindings: tuple[DrawerInvocationBinding, ...],
        read_source_stamp: Callable[[], PhysicalObservationStamp],
    ) -> None:
        if (
            simulation.num_envs != 1
            or simulation.physics_backend != "default"
            or simulation.get_robot(robot.uid) is not robot
        ):
            raise ValueError(
                "Drawer planning requires the exact Default/B1 robot owner."
            )
        if not bindings or not callable(read_source_stamp):
            raise ValueError(
                "Drawer planning requires explicit bindings and native source guard."
            )
        self._robot, self._simulation, self._read_stamp = (
            robot,
            simulation,
            read_source_stamp,
        )
        self._bindings = {}
        self._observations = {}
        self._native_owners = {}
        for binding in bindings:
            if type(binding) is not DrawerInvocationBinding:
                raise TypeError("Drawer bindings must be typed declarations.")
            rows = self._bindings.setdefault(binding.invocation_id, [])
            if rows and (
                binding.operation != "close"
                or any(
                    row.operation != "close" or row.object_id == binding.object_id
                    for row in rows
                )
            ):
                raise ValueError(
                    "Drawer occurrences must resolve one motion or distinct close payloads."
                )
            rows.append(binding)
            key = (binding.source, binding.object_id, binding.lower, binding.upper)
            if key not in self._observations:
                self._observations[key] = DrawerPayloadObservation(
                    simulation,
                    binding.source,
                    binding.object_id,
                    binding.lower,
                    binding.upper,
                )
                self._native_owners[key] = (
                    simulation.get_articulation(binding.source.object_id),
                    simulation.get_rigid_object(binding.object_id),
                )

    @property
    def robot(self) -> Robot:
        """Return the borrowed exact engine robot."""
        return self._robot

    def _guard(self, context: PlanningContext) -> None:
        stamp = context.world_source_stamp
        observed = (
            context.observed_robot
            if context.observed_robot is not None
            else context.robot
        )
        if (
            context.batch_size != 1
            or context.env_ids.tolist() != [0]
            or type(stamp) is not PhysicalObservationStamp
            or stamp.source_kind != "default.contacts"
            or self._read_stamp() != stamp
            or observed.root_pose is None
            or observed.timestamp != context.scene.timestamp
        ):
            raise ValueError(
                "Drawer planning requires its original complete native observation."
            )
        if not torch.equal(self.robot.get_qpos(), observed.qpos) or not torch.equal(
            self.robot.get_local_pose(to_matrix=True).to(observed.root_pose),
            observed.root_pose,
        ):
            raise ValueError(
                "Drawer native robot differs from its retained observation."
            )
        for key, (art, obj) in self._native_owners.items():
            source, object_id, _, _ = key
            if (
                self._simulation.get_articulation(source.object_id) is not art
                or self._simulation.get_rigid_object(object_id) is not obj
            ):
                raise ValueError("Drawer native entity ownership changed.")

    def inputs(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> tuple[DrawerPlanningInputs, ...]:
        """Return guarded local views for this exact declared execution occurrence."""
        self._guard(context)
        rows = self._bindings.get(request.invocation_id)
        expected = {"place": "place", "transport": "move_held_object", "close": "slide"}
        if rows is None or any(
            request.skill_id != expected[row.operation] for row in rows
        ):
            raise ValueError("Drawer occurrence/skill has no matching declaration.")
        views = {
            key: value.guarded(lambda: self._guard(context))
            for key, value in self._observations.items()
        }
        result = []
        for row in rows:
            key = (row.source, row.object_id, row.lower, row.upper)
            others = tuple(
                value
                for other_key, value in views.items()
                if other_key[1] != row.object_id
                and (other_key[0].object_id, other_key[0].link)
                == (row.source.object_id, row.source.link)
            )
            result.append(
                DrawerPlanningInputs(
                    row.target_id, row.link_entity_id, views[key], others
                )
            )
        return tuple(result)

    def prepare_articulation_query(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> RobotArticulationCollisionSnapshot:
        """Capture owned articulation-only geometry under the same native guard."""
        self.inputs(request, context)
        motion = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        query = RobotArticulationCollisionSnapshot(
            self.robot, self._simulation, control_part=motion.control_part
        )
        try:
            self._guard(context)
            return query
        except BaseException:
            query.close()
            raise


def _validate_drawer_program(compiled, manifest, policies, bindings):
    from embodichain.lab.sim.atomic_actions import (
        PlaceOptions,
        MoveHeldObjectOptions,
        SlideOptions,
    )

    declarations = {}
    for binding in bindings:
        declarations.setdefault(binding.invocation_id, []).append(binding)
    selected = {value.invocation_id: value.preset_id for value in policies}
    used = set()
    for segment in compiled.iter_segments():
        for item in segment.calls:
            identity = (
                f"{compiled.program_id}/{segment.segment_id}:{item.segment_call_index}"
            )
            linked = manifest.link_call(
                item.call, path=item.source_path, preset_id=selected.get(identity)
            )
            preset = manifest.robot_profile.presets[linked.preset_id]
            options = preset.action_option_templates.get(item.call.semantic_id)
            operation = (
                "place"
                if isinstance(options, PlaceOptions) and options.drawer_placement
                else (
                    "transport"
                    if isinstance(options, MoveHeldObjectOptions)
                    and options.drawer_transport
                    else (
                        "close"
                        if isinstance(options, SlideOptions)
                        and options.drawer_containment
                        else None
                    )
                )
            )
            rows = declarations.get(identity, ())
            if (
                item.call.semantic_id == "gen_sim.drawer_transport"
                and operation != "transport"
            ):
                raise ValueError(
                    "Drawer transport cannot use generic MoveHeldObject options."
                )
            if operation is None:
                if rows:
                    raise ValueError(
                        "Drawer declaration requires its selected typed skill option."
                    )
                continue
            if (
                not rows
                or segment.parallel_block is not None
                or identity not in selected
                or any(row.operation != operation for row in rows)
            ):
                raise ValueError(
                    "Drawer planning requires its exact serial policy-bound occurrence."
                )
            if operation in ("place", "transport") and len(rows) != 1:
                raise ValueError("Drawer motion must select exactly one payload.")
            if operation == "transport" and (
                preset.motion_policy.strategy != "motion_gen"
                or options.linear_segments
                or options.lift_clearance is not None
                or preset.recovery_policy.planning_recovery is not None
            ):
                raise ValueError(
                    "Drawer transport requires explicit direct motion_gen policy."
                )
            if operation == "transport":
                if item.call.semantic_id != "gen_sim.drawer_transport" or dict(
                    item.call.arguments
                ) != {"object": rows[0].object_id, "target": rows[0].target_id}:
                    raise ValueError(
                        "Drawer transport requires its exact effectless source call and object target."
                    )
            if operation == "close":
                arguments = dict(getattr(item.call, "arguments", {}))
                if (
                    item.call.semantic_id != "gen_sim.articulation_slide"
                    or arguments.get("state") != "closed"
                    or any(
                        arguments.get("object") != row.source.object_id for row in rows
                    )
                ):
                    raise ValueError(
                        "Drawer containment requires its exact source closing call."
                    )
            if operation == "place":
                target = getattr(item.call, "inside", None)
                obj = getattr(item.call, "object", None)
                if (
                    getattr(target, "entity_id", None) != rows[0].target_id
                    or getattr(obj, "entity_id", None) != rows[0].object_id
                ):
                    raise ValueError(
                        "Drawer Place requires its declared payload and inside target."
                    )
            used.add(identity)
    if used != set(declarations):
        raise ValueError("Drawer declarations contain unused or unknown occurrences.")
