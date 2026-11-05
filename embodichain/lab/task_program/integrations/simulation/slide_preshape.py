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

"""Explicit source prismatic hand sweep assembly and occurrence ownership."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING
import torch
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.sim.utility.grasp_geometry import RobotHandGeometryCfg
from embodichain.lab.sim.utility.preshape_geometry import (
    capture_preshape_hand_clearance,
)
from embodichain.lab.sim.scene.prismatic_control import measure_native_support_top
from embodichain.toolkits.graspkit.pg_grasp.preshape import PreshapeHandClearance

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager
    from embodichain.lab.sim.atomic_actions import (
        ResolvedActionRequest,
        PlanningContext,
    )
__all__ = ["SlidePreshapeBinding", "SimulationSlideHandClearanceProvider"]


@dataclass(frozen=True, slots=True)
class SlidePreshapeBinding:
    """Bind one source Slide occurrence to an exact calibrated hand and support."""

    invocation_id: str
    grasp_target_id: str
    table_uid: str

    def __post_init__(self) -> None:
        for name in ("invocation_id", "grasp_target_id", "table_uid"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError(f"{name} must be an exact identity.")


class SimulationSlideHandClearanceProvider:
    """Cache original hand sweeps once; measure bounded fractional sweeps late."""

    def __init__(
        self,
        robot: Robot,
        simulation: SimulationManager,
        *,
        bindings: tuple[SlidePreshapeBinding, ...],
        hand_geometry_cfgs: Mapping[str, RobotHandGeometryCfg],
        read_source_stamp: Callable[[], PhysicalObservationStamp],
    ) -> None:
        """Capture nominal seven-state sweeps during actual runtime assembly only.

        Args:
            robot: Exact engine robot owned by the Default/B1 simulation.
            simulation: Borrowed physical owner, never stepped or reset here.
            bindings: Exact prismatic occurrences and hand/support identities.
            hand_geometry_cfgs: Complete explicit hand calibration roster in
                original profile order. Every hand is measured for each table.
            read_source_stamp: Complete source/lifetime guard for all reads.
        """
        if (
            simulation.num_envs != 1
            or simulation.physics_backend != "default"
            or simulation.get_robot(robot.uid) is not robot
        ):
            raise ValueError(
                "Source prismatic hand sweeps require exact Default/B1 robot ownership."
            )
        if (
            type(bindings) is not tuple
            or not bindings
            or any(type(value) is not SlidePreshapeBinding for value in bindings)
        ):
            raise TypeError(
                "Slide preshape bindings must be a nonempty immutable tuple."
            )
        if len({value.invocation_id for value in bindings}) != len(bindings):
            raise ValueError("Slide preshape occurrences must be unique.")
        if not callable(read_source_stamp):
            raise TypeError("Slide preshape requires a native source-stamp guard.")
        cfgs = {key: deepcopy(value) for key, value in hand_geometry_cfgs.items()}
        for cfg in cfgs.values():
            cfg.validate_values()
        if any(value.grasp_target_id not in cfgs for value in bindings):
            raise ValueError("Slide preshape requires every selected calibrated hand.")
        self._robot, self._simulation = robot, simulation
        self._bindings = MappingProxyType(
            {value.invocation_id: value for value in bindings}
        )
        self._cfgs = MappingProxyType(cfgs)
        self._read_stamp = read_source_stamp
        self._tables = {
            uid: simulation.get_rigid_object(uid)
            for uid in dict.fromkeys(value.table_uid for value in bindings)
        }
        if any(value is None for value in self._tables.values()):
            raise ValueError("Slide preshape requires its declared native support.")
        self._nominal = {}
        for uid in self._tables:
            self._nominal.update(
                {
                    (key, uid): value
                    for key, value in self._capture_roster(uid, 0.85).items()
                }
            )

    @property
    def robot(self) -> Robot:
        """Return the borrowed exact engine robot."""
        return self._robot

    def _capture_roster(self, table_uid, fraction, expected_qpos=None):
        stamp = self._read_stamp()
        if (
            type(stamp) is not PhysicalObservationStamp
            or stamp.source_kind != "default.contacts"
        ):
            raise ValueError(
                "Preshape capture requires an explicit Default native stamp."
            )
        table = self._tables[table_uid]
        if self._simulation.get_rigid_object(table_uid) is not table:
            raise ValueError("Preshape support owner changed.")
        table_z = measure_native_support_top(table)
        result = {}
        for key, cfg in self._cfgs.items():
            observed = (
                self.robot.get_qpos().clone()
                if expected_qpos is None
                else expected_qpos
            )
            result[key] = capture_preshape_hand_clearance(
                self.robot,
                cfg=cfg,
                expected_qpos=observed,
                table_z=table_z,
                preshape_fraction=fraction,
            )
        if self._read_stamp() != stamp:
            raise ValueError("Preshape native source changed during capture.")
        return result

    def _binding(self, request, context):
        declaration = self._bindings.get(request.invocation_id)
        if (
            declaration is None
            or request.skill_id != "slide"
            or context.batch_size != 1
            or context.env_ids.tolist() != [0]
        ):
            raise ValueError(
                "Preshape lookup requires a declared B1/env0 Slide occurrence."
            )
        stamp = context.world_source_stamp
        if (
            type(stamp) is not PhysicalObservationStamp
            or stamp.source_kind != "default.contacts"
            or self._read_stamp() != stamp
        ):
            raise ValueError(
                "Preshape requires its original complete native observation."
            )
        hand = request.binding.endpoint("primary", "grasp").require_target(
            JointPositionTarget
        )
        arm = request.binding.endpoint("primary", "motion").require_target(
            JointPositionTarget
        )
        cfg = self._cfgs[declaration.grasp_target_id]
        if (
            hand.target_id != declaration.grasp_target_id
            or hand.control_part != cfg.hand_part
            or arm.control_part != cfg.motion_part
        ):
            raise ValueError(
                "Preshape hand/motion resources differ from the declared calibration."
            )
        if (
            self._simulation.get_rigid_object(declaration.table_uid)
            is not self._tables[declaration.table_uid]
        ):
            raise ValueError("Preshape support owner changed.")
        return declaration

    def nominal(
        self, request: ResolvedActionRequest, context: PlanningContext
    ) -> PreshapeHandClearance:
        """Return an owned nominal snapshot without new FK or geometry reads.

        Args:
            request: Exact invocation and resolved hand/motion endpoint binding.
            context: Matching complete native observation and environment row.

        Returns:
            Independent source-compatible numeric hand sweep.
        """
        binding = self._binding(request, context)
        result = self._nominal[binding.grasp_target_id, binding.table_uid]
        return replace(result, points=result.points.clone())

    def measure(
        self, request: ResolvedActionRequest, context: PlanningContext, fraction: float
    ) -> PreshapeHandClearance:
        """Measure the original full hand roster for one late aperture candidate.

        Args:
            request: Exact declared Slide invocation and resource binding.
            context: Original measured robot and native source observation.
            fraction: One source closure fraction within 0.75--0.85.

        Returns:
            Selected hand sweep after every profile hand is captured in order.
        """
        binding = self._binding(request, context)
        observed = (
            context.observed_robot
            if context.observed_robot is not None
            else context.robot
        )
        result = self._capture_roster(binding.table_uid, fraction, observed.qpos)[
            binding.grasp_target_id
        ]
        self._binding(request, context)
        return result


def _validate_slide_preshape_program(compiled, manifest, policies, bindings):
    from embodichain.lab.sim.atomic_actions.primitives.slide import SlideOptions
    from embodichain.lab.sim.atomic_actions.articulation_continuation import (
        SlidePlanningRecovery,
    )

    if (
        not bindings
        and not any(
            isinstance(options, SlideOptions) and options.qualify_hand_clearance
            for preset in manifest.robot_profile.presets.values()
            for options in preset.action_option_templates.values()
        )
        and not any(
            isinstance(preset.recovery_policy.planning_recovery, SlidePlanningRecovery)
            and preset.recovery_policy.planning_recovery.preshape_aperture_recovery
            for preset in manifest.robot_profile.presets.values()
        )
    ):
        return
    selected = {value.invocation_id: value.preset_id for value in policies}
    declared = {value.invocation_id: value for value in bindings}
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
            rule = preset.recovery_policy.planning_recovery
            recovery = (
                isinstance(rule, SlidePlanningRecovery)
                and rule.preshape_aperture_recovery
            )
            qualify = (
                isinstance(options, SlideOptions) and options.qualify_hand_clearance
            )
            if not qualify:
                if identity in declared or recovery:
                    raise ValueError(
                        "Preshape declaration/recovery requires selected Slide hand-clearance options."
                    )
                continue
            if (
                identity not in declared
                or segment.parallel_block is not None
                or options.preshape_fraction != 0.85
                or preset.motion_policy.strategy != "ik_interp"
            ):
                raise ValueError(
                    "Preshape requires its exact serial occurrence and original85% ik_interp policy."
                )
            binding = declared[identity]
            default = manifest.robot_profile.defaults.get("slide")
            resource = item.call.resources.get(
                "primary", None if default is None else default.resources.get("primary")
            )
            declaration = manifest.robot_profile.resources.get(resource)
            hand = None if declaration is None else declaration.endpoints.get("grasp")
            if getattr(hand, "control_part", None) != binding.grasp_target_id:
                raise ValueError(
                    "Preshape selected resource differs from its declared calibrated hand."
                )
            used.add(identity)
    if used != set(declared):
        raise ValueError("Preshape declarations contain unused/unknown occurrences.")
