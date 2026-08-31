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

"""Simulation-backed Task Program post-policies and validators.

The port in this module deliberately consumes the same explicit
:class:`SimulationSceneBinding` used to construct the semantic scene registry.
It never scans a simulation or guesses a native entity from a canonical name.
Post-policy actions remain inside the normal Gym ``env.step()`` path owned by
:class:`TaskProgramDemoBridge`. Rows eligible for the policy reuse full drive targets
so physical contact does not erase position-control preload; rows already
inactive use fresh measured-position holds.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

import torch

from embodichain.lab.gym.envs.settling import (
    DynamicSettleMonitor,
    DynamicSettleMonitorCfg,
    DynamicSettleSample,
    DynamicSettleState,
)

from embodichain.lab.task_program.compiler import (
    CompiledArticulationJointPositionValidator,
    CompiledObjectNearTargetValidator,
    CompiledPostPolicy,
    CompiledTaskProgramSegment,
    CompiledTaskProgramValidator,
)
from .bindings import SimulationSceneBinding

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot
    from embodichain.lab.sim.sim_manager import SimulationManager


@dataclass(frozen=True, slots=True)
class _SimulationSettleTarget:
    """One canonical entity resolved through an explicit native binding."""

    canonical_id: str
    kind: str
    native_entity: Any


def default_simulation_settle_presets() -> Mapping[str, DynamicSettleMonitorCfg]:
    """Return independently owned built-in post-policy presets."""
    return MappingProxyType(
        {
            "rigid_object": DynamicSettleMonitorCfg(
                linear_velocity_threshold=0.03,
                angular_velocity_threshold=0.20,
                min_steps=10,
                max_steps=240,
                check_interval_steps=2,
                required_stable_checks=3,
            ),
            "articulation": DynamicSettleMonitorCfg(
                linear_velocity_threshold=0.02,
                angular_velocity_threshold=0.10,
                min_steps=10,
                max_steps=120,
                check_interval_steps=2,
                required_stable_checks=3,
            ),
        }
    )


def _json_speed_values(value: torch.Tensor) -> list[float | None]:
    """Convert speed evidence to finite JSON numbers or explicit unknowns."""
    return [
        float(item) if math.isfinite(float(item)) else None
        for item in value.detach().cpu().tolist()
    ]


class SimulationSegmentPolicyPort:
    """Execute built-in segment policies against explicitly bound simulation data.

    Args:
        simulation: Live simulation used only for UIDs declared in
            ``scene_binding``.
        robot: Live robot used to produce full target-qpos holds while the
            post-policy observes settling.
        scene_binding: Exact canonical-to-native scene declaration.
        settle_presets: Named settling policies. ``None`` installs the shared
            ``rigid_object`` and ``articulation`` presets.
        env_ids: Optional stable logical row IDs. They describe correlation,
            not simulator row indices; simulator rows remain ordered exactly as
            returned by the robot and bound entities.

    The same instance implements both ``SegmentPostPolicyPort`` and
    ``SegmentValidatorPort``. Unknown policy types, presets, canonical IDs, or
    native entities fail before an action is emitted.
    """

    def __init__(
        self,
        simulation: SimulationManager,
        robot: Robot,
        scene_binding: SimulationSceneBinding,
        *,
        settle_presets: Mapping[str, DynamicSettleMonitorCfg] | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        if type(scene_binding) is not SimulationSceneBinding:
            raise TypeError("scene_binding must be exactly SimulationSceneBinding.")
        qpos = self._read_robot_qpos(robot, target=False)
        self._robot_qpos_shape = qpos.shape
        self._robot_qpos_device = qpos.device
        if env_ids is None:
            env_ids = torch.arange(
                qpos.shape[0],
                dtype=torch.long,
                device=qpos.device,
            )
        if not isinstance(env_ids, torch.Tensor):
            raise TypeError("env_ids must be a torch.Tensor or None.")
        if env_ids.dtype != torch.long or env_ids.shape != (qpos.shape[0],):
            raise ValueError("env_ids must be int64 with one ID per simulator row.")
        if env_ids.device != qpos.device:
            raise ValueError("env_ids and robot qpos must share a device.")
        if torch.unique(env_ids).numel() != env_ids.numel():
            raise ValueError("env_ids must contain unique values.")

        selected_presets = (
            default_simulation_settle_presets()
            if settle_presets is None
            else settle_presets
        )
        if not isinstance(selected_presets, Mapping) or not selected_presets:
            raise ValueError("settle_presets must be a non-empty mapping.")
        normalized_presets: dict[str, DynamicSettleMonitorCfg] = {}
        for preset_id, cfg in selected_presets.items():
            if (
                type(preset_id) is not str
                or not preset_id
                or preset_id != preset_id.strip()
            ):
                raise ValueError(
                    "Settle preset IDs must be non-empty strings without outer "
                    "whitespace."
                )
            if not isinstance(cfg, DynamicSettleMonitorCfg):
                raise TypeError(
                    "settle_presets values must be DynamicSettleMonitorCfg values."
                )
            normalized_presets[preset_id] = cfg.snapshot()

        self._simulation = simulation
        self._robot = robot
        self._scene_binding = scene_binding
        self._env_ids = env_ids.clone()
        self._row_indices = torch.arange(
            qpos.shape[0],
            dtype=torch.long,
            device=qpos.device,
        )
        self._settle_presets = MappingProxyType(normalized_presets)
        (
            self._settle_targets,
            self._rigid_objects,
            self._articulations,
        ) = self._resolve_native_entities()
        self._post_policy_results: dict[int, dict[str, object]] = {}
        self._post_policy_success: dict[int, torch.Tensor] = {}
        self._validator_results: dict[int, dict[str, object]] = {}

    @property
    def settle_preset_ids(self) -> tuple[str, ...]:
        """Return installed post-policy preset IDs in declaration order."""
        return tuple(self._settle_presets)

    def validate_policy(
        self,
        policy: Any,
        *,
        segment: Any,
    ) -> None:
        """Validate one post-policy against static bindings without observation.

        This method reads only the compiled declaration, installed preset
        table, and entities resolved when the port was constructed. It never
        samples velocity or qpos and never emits a controller action.
        """
        if type(policy) is not CompiledPostPolicy:
            raise TypeError("policy must be exactly CompiledPostPolicy.")
        self._validate_segment_membership(segment, policy, kind="post policy")
        if policy.cfg.kind != "wait_stable":
            raise ValueError(
                f"Unsupported compiled post-policy kind {policy.cfg.kind!r}."
            )
        if policy.cfg.preset not in self._settle_presets:
            raise KeyError(
                f"Unknown settle preset {policy.cfg.preset!r}; available presets "
                f"are {sorted(self._settle_presets)}."
            )
        entity_id = policy.entity.entity_id
        target = self._settle_targets.get(entity_id)
        if target is None:
            raise KeyError(
                f"Canonical settle entity {entity_id!r} has no explicit native "
                "dynamic binding."
            )
        if target.kind == "rigid_object" and bool(
            getattr(target.native_entity, "is_non_dynamic", False)
        ):
            raise ValueError(
                f"Canonical settle entity {entity_id!r} is static or kinematic."
            )

    def actions(
        self,
        policy: Any,
        *,
        segment: Any,
        active_mask: torch.Tensor,
    ) -> Iterator[torch.Tensor]:
        """Yield full target-qpos hold actions until rows settle or time out.

        Args:
            policy: Exact compiled ``wait_stable`` policy.
            segment: Exact segment that owns ``policy``.
            active_mask: Rows that remain eligible after runtime execution and
                preceding post-policies. Inactive rows are held safely but do
                not participate in settling, timeout, or success results.

        Yields:
            Fresh full target-qpos hold commands consumed by ordinary
            ``env.step()``. Reading drive targets instead of measured joint
            positions preserves contact preload in position-controlled tools.
            Rows inactive when the policy starts use fresh measured qpos holds.

        Timeout is a normal row-local result boundary. Timed-out rows are
        exposed through :meth:`post_policy_result` and
        :meth:`post_policy_metadata`; no batch-level exception is raised.
        """
        self.validate_policy(policy, segment=segment)
        active_mask = self._validate_active_mask(active_mask)
        preset = self._settle_presets[policy.cfg.preset]
        entity_id = policy.entity.entity_id
        target = self._settle_targets[entity_id]

        result_key = id(policy)
        self._post_policy_results.pop(result_key, None)
        self._post_policy_success.pop(result_key, None)
        if not bool(active_mask.any().item()):
            self._post_policy_success[result_key] = active_mask.clone()
            self._post_policy_results[result_key] = {
                "kind": policy.cfg.kind,
                "entity_id": entity_id,
                "preset": policy.cfg.preset,
                "source_path": list(policy.source_path),
                "status": "skipped",
                "active_mask": active_mask.detach().cpu().tolist(),
                "thresholds": self._settle_threshold_metadata(preset),
                "state": self._empty_settle_state_metadata(active_mask),
            }
            return

        active_rows = self._row_indices[active_mask]
        monitor = DynamicSettleMonitor(preset, self._env_ids[active_mask])
        elapsed_steps = 0
        while True:
            state = monitor.observe(
                (self._measure_settle_target(target, row_indices=active_rows),),
                elapsed_steps=elapsed_steps,
            )
            settled_mask = torch.zeros_like(active_mask)
            settled_mask[active_mask] = state.settled_mask
            self._post_policy_results[result_key] = {
                "kind": policy.cfg.kind,
                "entity_id": entity_id,
                "preset": policy.cfg.preset,
                "source_path": list(policy.source_path),
                "active_mask": active_mask.detach().cpu().tolist(),
                "status": (
                    "settled"
                    if bool(state.settled_mask.all().item())
                    else (
                        "timed_out"
                        if bool(state.timeout_mask.any().item())
                        else "running"
                    )
                ),
                "thresholds": self._settle_threshold_metadata(preset),
                "state": self._expand_settle_state_metadata(state, active_mask),
            }
            self._post_policy_success[result_key] = settled_mask
            if bool(state.settled_mask.all().item()):
                return
            if bool(state.timeout_mask.any().item()):
                return
            yield self._hold_robot_qpos(active_mask)
            elapsed_steps += 1

    def post_policy_result(
        self,
        policy: Any,
        *,
        segment: Any,
    ) -> torch.Tensor:
        """Return the latest independently owned per-row settling result."""
        if type(policy) is not CompiledPostPolicy:
            raise TypeError("policy must be exactly CompiledPostPolicy.")
        self._validate_segment_membership(segment, policy, kind="post policy")
        result = self._post_policy_success.get(id(policy))
        if result is None:
            raise RuntimeError("Post-policy result is unavailable before execution.")
        return result.clone()

    def post_policy_metadata(
        self,
        policy: Any,
        *,
        segment: Any,
    ) -> Mapping[str, object]:
        """Return the latest JSON-safe settling trace for one policy.

        The trace is available after the policy generator has started. A
        terminal trace has status ``"settled"``, ``"timed_out"``, or
        ``"skipped"`` when no rows remain active; an early demo interruption
        intentionally retains the latest ``"running"`` snapshot for diagnosis.
        """
        if type(policy) is not CompiledPostPolicy:
            raise TypeError("policy must be exactly CompiledPostPolicy.")
        self._validate_segment_membership(segment, policy, kind="post policy")
        metadata = self._post_policy_results.get(id(policy))
        if metadata is None:
            raise RuntimeError("Post-policy metadata is unavailable before execution.")
        return deepcopy(metadata)

    def validate_validator(
        self,
        validator: Any,
        *,
        segment: Any,
    ) -> None:
        """Validate one validator against static bindings without observation."""
        if type(validator) not in (
            CompiledObjectNearTargetValidator,
            CompiledArticulationJointPositionValidator,
        ):
            raise TypeError("validator must be an exact compiled validator.")
        self._validate_segment_membership(segment, validator, kind="validator")
        if type(validator) is CompiledObjectNearTargetValidator:
            if validator.cfg.kind != "object_near_target":
                raise ValueError(
                    f"Unsupported compiled validator kind {validator.cfg.kind!r}."
                )
            entity_id = validator.object.entity_id
            if entity_id not in self._rigid_objects:
                raise KeyError(
                    f"Canonical validator object {entity_id!r} has no explicit "
                    "rigid-object binding."
                )
            return

        if validator.cfg.kind != "articulation_joint_position":
            raise ValueError(
                f"Unsupported compiled validator kind {validator.cfg.kind!r}."
            )
        entity_id = validator.articulation.entity_id
        articulation = self._articulations.get(entity_id)
        if articulation is None:
            raise KeyError(
                f"Canonical validator articulation {entity_id!r} has no explicit "
                "articulation binding."
            )
        self._articulation_joint_index(
            articulation,
            entity_id=entity_id,
            joint_name=validator.cfg.joint,
        )

    def validate(self, validator: Any, *, segment: Any) -> torch.Tensor:
        """Observe one explicitly bound entity against its validator contract.

        Args:
            validator: Exact compiled segment validator.
            segment: Exact segment that owns ``validator``.

        Returns:
            Boolean tensor with one result per simulation row.
        """
        self.validate_validator(validator, segment=segment)
        if type(validator) is CompiledArticulationJointPositionValidator:
            return self._validate_articulation_joint_position(validator)
        if type(validator) is not CompiledObjectNearTargetValidator:
            raise TypeError("validator must be an exact compiled validator.")
        entity_id = validator.object.entity_id
        entity = self._rigid_objects[entity_id]
        pose = self._read_pose(entity, entity_id=entity_id)
        current_position = pose[:, :3, 3]
        target_position = validator.target_pose.position.to(
            device=current_position.device,
            dtype=current_position.dtype,
        )
        if target_position.dim() == 1:
            target_position = target_position.unsqueeze(0).expand_as(current_position)
        elif target_position.shape != current_position.shape:
            raise ValueError(
                "Validator target batch must be unbatched or match simulator rows."
            )
        error = torch.linalg.vector_norm(current_position - target_position, dim=1)
        accepted = torch.isfinite(error) & (
            error <= float(validator.cfg.position_tolerance)
        )
        self._validator_results[id(validator)] = {
            "kind": validator.cfg.kind,
            "object_id": entity_id,
            "target_id": validator.target_selection.target_id,
            "target_value_index": validator.target_selection.value_index,
            "source_path": list(validator.source_path),
            "position_tolerance": float(validator.cfg.position_tolerance),
            "env_ids": self._env_ids.detach().cpu().tolist(),
            "object_position": current_position.detach().cpu().tolist(),
            "target_position": target_position.detach().cpu().tolist(),
            "position_error": error.detach().cpu().tolist(),
            "accepted_mask": accepted.detach().cpu().tolist(),
        }
        return accepted

    def validator_metadata(
        self,
        validator: Any,
        *,
        segment: Any,
    ) -> Mapping[str, object]:
        """Return an owned JSON-safe trace for one completed validator."""
        if type(validator) not in (
            CompiledObjectNearTargetValidator,
            CompiledArticulationJointPositionValidator,
        ):
            raise TypeError("validator must be an exact compiled validator.")
        self._validate_segment_membership(segment, validator, kind="validator")
        metadata = self._validator_results.get(id(validator))
        if metadata is None:
            raise RuntimeError("Validator metadata is unavailable before validation.")
        return deepcopy(metadata)

    def _validate_articulation_joint_position(
        self,
        validator: CompiledArticulationJointPositionValidator,
    ) -> torch.Tensor:
        """Evaluate one measured articulation joint against inclusive bounds."""
        entity_id = validator.articulation.entity_id
        articulation = self._articulations[entity_id]
        joint_index = self._articulation_joint_index(
            articulation,
            entity_id=entity_id,
            joint_name=validator.cfg.joint,
        )
        qpos = self._read_articulation_qpos(articulation, entity_id=entity_id)
        if joint_index >= qpos.shape[1]:
            raise ValueError(
                f"Joint {validator.cfg.joint!r} resolves to index {joint_index}, "
                f"outside articulation {entity_id!r} qpos width {qpos.shape[1]}."
            )
        position = qpos[:, joint_index]
        accepted = torch.isfinite(position)
        if validator.cfg.minimum_position is not None:
            accepted &= position >= float(validator.cfg.minimum_position)
        if validator.cfg.maximum_position is not None:
            accepted &= position <= float(validator.cfg.maximum_position)
        self._validator_results[id(validator)] = {
            "kind": validator.cfg.kind,
            "articulation_id": entity_id,
            "joint": validator.cfg.joint,
            "source_path": list(validator.source_path),
            "minimum_position": validator.cfg.minimum_position,
            "maximum_position": validator.cfg.maximum_position,
            "env_ids": self._env_ids.detach().cpu().tolist(),
            "joint_position": [
                float(value) if math.isfinite(float(value)) else None
                for value in position.detach().cpu().tolist()
            ],
            "accepted_mask": accepted.detach().cpu().tolist(),
        }
        return accepted

    @staticmethod
    def _read_robot_qpos(robot: Robot, *, target: bool) -> torch.Tensor:
        """Capture one finite current- or target-qpos full-robot batch."""
        if type(target) is not bool:
            raise TypeError("target must be a bool.")
        mode = "target" if target else "current"
        call = f"robot.get_qpos(target={target})"
        get_qpos = getattr(robot, "get_qpos", None)
        if not callable(get_qpos):
            raise TypeError(f"robot must provide {call}.")
        qpos = get_qpos(target=target)
        if (
            not isinstance(qpos, torch.Tensor)
            or not qpos.is_floating_point()
            or qpos.dim() != 2
            or qpos.shape[0] == 0
            or qpos.shape[1] == 0
        ):
            raise ValueError(
                f"{call} must return {mode} floating full-qpos shape (B, J)."
            )
        if not bool(torch.isfinite(qpos).all().item()):
            raise ValueError(f"{call} must return finite {mode} qpos values.")
        return qpos.clone()

    def _hold_robot_qpos(self, active_mask: torch.Tensor) -> torch.Tensor:
        """Keep initial active rows on targets and inactive rows on current qpos."""
        target_qpos = self._read_robot_qpos(self._robot, target=True)
        if (
            target_qpos.shape != self._robot_qpos_shape
            or target_qpos.device != self._robot_qpos_device
        ):
            raise ValueError(
                "robot.get_qpos(target=True) target full qpos must match the "
                "construction-time current full qpos shape and device."
            )
        if bool(active_mask.all().item()):
            return target_qpos

        current_qpos = self._read_robot_qpos(self._robot, target=False)
        if (
            current_qpos.shape != self._robot_qpos_shape
            or current_qpos.device != self._robot_qpos_device
        ):
            raise ValueError(
                "robot.get_qpos(target=False) current full qpos must match the "
                "construction-time current full qpos shape and device."
            )
        hold_qpos = current_qpos.clone()
        hold_qpos[active_mask] = target_qpos[active_mask]
        return hold_qpos

    def _validate_active_mask(self, active_mask: torch.Tensor) -> torch.Tensor:
        """Return one owned row mask aligned with the simulator batch."""
        if not isinstance(active_mask, torch.Tensor):
            raise TypeError("active_mask must be a torch.Tensor.")
        if active_mask.dtype != torch.bool or active_mask.shape != self._env_ids.shape:
            raise ValueError(
                "active_mask must be bool with one value per simulator row."
            )
        if active_mask.device != self._env_ids.device:
            raise ValueError("active_mask and env_ids must share a device.")
        return active_mask.clone()

    @staticmethod
    def _settle_threshold_metadata(
        preset: DynamicSettleMonitorCfg,
    ) -> dict[str, float | int]:
        """Serialize one settling preset without exposing mutable state."""
        return {
            "linear_velocity": float(preset.linear_velocity_threshold),
            "angular_velocity": float(preset.angular_velocity_threshold),
            "min_steps": preset.min_steps,
            "max_steps": preset.max_steps,
            "check_interval_steps": preset.check_interval_steps,
            "required_stable_checks": preset.required_stable_checks,
        }

    def _empty_settle_state_metadata(
        self,
        active_mask: torch.Tensor,
    ) -> dict[str, object]:
        """Return a full-batch trace for a policy with no eligible rows."""
        batch_size = self._env_ids.numel()
        return {
            "elapsed_steps": 0,
            "observation_count": 0,
            "env_ids": self._env_ids.detach().cpu().tolist(),
            "active_mask": active_mask.detach().cpu().tolist(),
            "stable_counts": [0] * batch_size,
            "settled_mask": [False] * batch_size,
            "timeout_mask": [False] * batch_size,
            "checked": False,
            "max_linear_speed": [None] * batch_size,
            "max_angular_speed": [None] * batch_size,
        }

    def _expand_settle_state_metadata(
        self,
        state: DynamicSettleState,
        active_mask: torch.Tensor,
    ) -> dict[str, object]:
        """Expand active-row monitor state to the stable full-batch ordering."""
        stable_counts = torch.zeros_like(self._env_ids)
        settled_mask = torch.zeros_like(active_mask)
        timeout_mask = torch.zeros_like(active_mask)
        max_linear_speed = torch.full(
            active_mask.shape,
            float("inf"),
            dtype=state.max_linear_speed.dtype,
            device=active_mask.device,
        )
        max_angular_speed = torch.full_like(max_linear_speed, float("inf"))
        stable_counts[active_mask] = state.stable_counts
        settled_mask[active_mask] = state.settled_mask
        timeout_mask[active_mask] = state.timeout_mask
        max_linear_speed[active_mask] = state.max_linear_speed
        max_angular_speed[active_mask] = state.max_angular_speed
        return {
            "elapsed_steps": state.elapsed_steps,
            "observation_count": state.observation_count,
            "env_ids": self._env_ids.detach().cpu().tolist(),
            "active_mask": active_mask.detach().cpu().tolist(),
            "stable_counts": stable_counts.detach().cpu().tolist(),
            "settled_mask": settled_mask.detach().cpu().tolist(),
            "timeout_mask": timeout_mask.detach().cpu().tolist(),
            "checked": state.checked,
            "max_linear_speed": _json_speed_values(max_linear_speed),
            "max_angular_speed": _json_speed_values(max_angular_speed),
        }

    @staticmethod
    def _validate_segment_membership(
        segment: Any,
        member: CompiledPostPolicy | CompiledTaskProgramValidator,
        *,
        kind: str,
    ) -> None:
        """Require the supplied compiled value to belong to the exact segment."""
        if type(segment) is not CompiledTaskProgramSegment:
            raise TypeError("segment must be exactly CompiledTaskProgramSegment.")
        values = (
            segment.post_policies
            if type(member) is CompiledPostPolicy
            else segment.validators
        )
        if not any(value is member for value in values):
            raise ValueError(
                f"Compiled {kind} does not belong to the supplied segment."
            )

    def _resolve_native_entities(
        self,
    ) -> tuple[
        Mapping[str, _SimulationSettleTarget],
        Mapping[str, Any],
        Mapping[str, Any],
    ]:
        """Resolve only explicitly declared canonical/native pairs."""
        settle_targets: dict[str, _SimulationSettleTarget] = {}
        rigid_objects: dict[str, Any] = {}
        articulation_targets: dict[str, _SimulationSettleTarget] = {}

        for binding in self._scene_binding.rigid_objects:
            entity = self._require_native(
                "get_rigid_object",
                canonical_id=binding.entity_id,
                simulation_uid=binding.simulation_uid,
            )
            target = _SimulationSettleTarget(
                binding.entity_id,
                "rigid_object",
                entity,
            )
            settle_targets[binding.entity_id] = target
            rigid_objects[binding.entity_id] = entity

        for binding in self._scene_binding.articulations:
            entity = self._require_native(
                "get_articulation",
                canonical_id=binding.entity_id,
                simulation_uid=binding.simulation_uid,
            )
            target = _SimulationSettleTarget(
                binding.entity_id,
                "articulation",
                entity,
            )
            settle_targets[binding.entity_id] = target
            articulation_targets[binding.entity_id] = target

        for binding in self._scene_binding.links:
            settle_targets[binding.entity_id] = self._require_parent_target(
                articulation_targets,
                child_id=binding.entity_id,
                parent_id=binding.articulation_id,
            )
        for binding in self._scene_binding.antipodal_grasps:
            parent = settle_targets.get(binding.object_id)
            if parent is None or parent.kind != "rigid_object":
                raise KeyError(
                    f"Affordance {binding.entity_id!r} references unavailable rigid "
                    f"object {binding.object_id!r}."
                )
            settle_targets[binding.entity_id] = parent
        articulations = {
            entity_id: target.native_entity
            for entity_id, target in articulation_targets.items()
        }
        return (
            MappingProxyType(settle_targets),
            MappingProxyType(rigid_objects),
            MappingProxyType(articulations),
        )

    @staticmethod
    def _require_parent_target(
        targets: Mapping[str, _SimulationSettleTarget],
        *,
        child_id: str,
        parent_id: str,
    ) -> _SimulationSettleTarget:
        """Resolve a child to one explicitly declared articulation root."""
        target = targets.get(parent_id)
        if target is None:
            raise KeyError(
                f"Canonical entity {child_id!r} references unavailable parent "
                f"{parent_id!r}."
            )
        return target

    def _require_native(
        self,
        getter_name: str,
        *,
        canonical_id: str,
        simulation_uid: str,
    ) -> Any:
        """Resolve one explicitly selected native simulation entity."""
        getter = getattr(self._simulation, getter_name, None)
        if not callable(getter):
            raise TypeError(f"simulation must provide {getter_name}().")
        entity = getter(simulation_uid)
        if entity is None:
            raise KeyError(
                f"Native entity {simulation_uid!r} selected for canonical entity "
                f"{canonical_id!r} was not found."
            )
        return entity

    def _measure_settle_target(
        self,
        target: _SimulationSettleTarget,
        *,
        row_indices: torch.Tensor,
    ) -> DynamicSettleSample:
        """Measure physical bodies for explicitly selected simulator rows."""
        if target.kind == "articulation":
            body_data = getattr(target.native_entity, "body_data", None)
            velocity = getattr(body_data, "body_link_vel", None)
            if not isinstance(velocity, torch.Tensor):
                raise RuntimeError(
                    f"Articulation settle target {target.canonical_id!r} has no "
                    "body_link_vel tensor."
                )
            selected = velocity.index_select(0, row_indices.to(velocity.device))
            if selected.dim() != 3 or selected.shape[-1] != 6:
                raise ValueError(
                    "Articulation body_link_vel must have shape (B, N, 6)."
                )
            linear_velocity = selected[..., :3]
            angular_velocity = selected[..., 3:]
        else:
            body_data = getattr(target.native_entity, "body_data", None)
            linear_velocity = getattr(body_data, "lin_vel", None)
            angular_velocity = getattr(body_data, "ang_vel", None)
            if not isinstance(linear_velocity, torch.Tensor) or not isinstance(
                angular_velocity,
                torch.Tensor,
            ):
                raise RuntimeError(
                    f"Rigid settle target {target.canonical_id!r} has no linear/"
                    "angular velocity tensors."
                )
            rows = row_indices.to(linear_velocity.device)
            linear_velocity = linear_velocity.index_select(0, rows)
            angular_velocity = angular_velocity.index_select(
                0,
                row_indices.to(angular_velocity.device),
            )
            if (
                linear_velocity.shape != angular_velocity.shape
                or linear_velocity.dim() < 2
                or linear_velocity.shape[-1] != 3
            ):
                raise ValueError(
                    "Rigid body velocities must have equal shape (B, ..., 3)."
                )

        linear_speed = torch.linalg.vector_norm(linear_velocity, dim=-1).reshape(
            row_indices.numel(),
            -1,
        )
        angular_speed = torch.linalg.vector_norm(angular_velocity, dim=-1).reshape(
            row_indices.numel(),
            -1,
        )
        device = self._env_ids.device
        return DynamicSettleSample(
            entity_id=target.canonical_id,
            linear_speed=linear_speed.to(device=device),
            angular_speed=angular_speed.to(device=device),
        )

    @staticmethod
    def _articulation_joint_index(
        articulation: Any,
        *,
        entity_id: str,
        joint_name: str,
    ) -> int:
        """Resolve one explicitly named native articulation joint."""
        names = getattr(articulation, "joint_names", None)
        if names is None or isinstance(names, (str, bytes)):
            raise TypeError(
                f"Articulation {entity_id!r} must expose an iterable joint_names."
            )
        try:
            normalized = tuple(names)
        except TypeError as exc:
            raise TypeError(
                f"Articulation {entity_id!r} must expose an iterable joint_names."
            ) from exc
        if not all(type(name) is str and name for name in normalized):
            raise ValueError(
                f"Articulation {entity_id!r} joint_names must be non-empty strings."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Articulation {entity_id!r} joint_names must be unique.")
        try:
            return normalized.index(joint_name)
        except ValueError as exc:
            raise KeyError(
                f"Joint {joint_name!r} was not found on articulation "
                f"{entity_id!r}; available joints are {sorted(normalized)}."
            ) from exc

    def _read_articulation_qpos(
        self,
        articulation: Any,
        *,
        entity_id: str,
    ) -> torch.Tensor:
        """Read one articulation position batch in simulator row order."""
        getter = getattr(articulation, "get_qpos", None)
        if not callable(getter):
            raise TypeError(
                f"Native articulation for {entity_id!r} must provide get_qpos()."
            )
        qpos = getter()
        if not isinstance(qpos, torch.Tensor) or not qpos.is_floating_point():
            raise TypeError("Articulation get_qpos() must return a floating tensor.")
        batch_size = int(self._env_ids.numel())
        if qpos.dim() != 2 or qpos.shape[0] != batch_size or qpos.shape[1] == 0:
            raise ValueError(
                f"Articulation {entity_id!r} qpos must have shape "
                f"({batch_size}, J) with J > 0."
            )
        return qpos.clone()

    def _read_pose(self, entity: Any, *, entity_id: str) -> torch.Tensor:
        """Read one rigid-object pose batch in simulator row order."""
        getter = getattr(entity, "get_local_pose", None)
        if not callable(getter):
            raise TypeError(
                f"Native rigid object for {entity_id!r} must provide "
                "get_local_pose()."
            )
        pose = getter(to_matrix=True)
        if not isinstance(pose, torch.Tensor) or not pose.is_floating_point():
            raise TypeError("get_local_pose(to_matrix=True) must return a tensor.")
        batch_size = int(self._env_ids.numel())
        if pose.shape == (4, 4):
            pose = pose.unsqueeze(0).expand(batch_size, -1, -1)
        elif pose.shape != (batch_size, 4, 4):
            raise ValueError(
                f"Rigid object {entity_id!r} pose must have shape "
                f"({batch_size}, 4, 4)."
            )
        if not bool(torch.isfinite(pose).all().item()):
            raise ValueError(f"Rigid object {entity_id!r} pose must be finite.")
        return pose.clone()


__all__: list[str] = []
