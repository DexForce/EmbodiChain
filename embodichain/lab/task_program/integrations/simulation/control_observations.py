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

"""Read actual completed control owners into plain physical observation packets."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, fields
import math
from typing import Any, TYPE_CHECKING
from .control_observation import ControlObservationSnapshot, ControlPolicyObservation

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
    from embodichain.lab.sim.evidence import PhysicalEvidenceFrame

__all__ = ["SimulationControlObservationCollector"]


def _diagnostic(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _diagnostic(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        return None
    return value


def _frame(frame: PhysicalEvidenceFrame, *, kind: str) -> dict[str, Any]:
    return dict(
        timestamp=frame.timestamp,
        observation_revision=frame.observation_revision,
        env_ids=frame.env_ids.tolist(),
        clock_domain="simulation_time",
        revision_kind=kind,
        batches={
            name: dict(
                evidence_id=batch.evidence_id,
                value_dtype=str(batch.values.dtype),
                value_shape=list(batch.values.shape),
                values=[
                    value.tolist() if valid else None
                    for value, valid in zip(
                        batch.values, batch.valid.tolist(), strict=True
                    )
                ],
                valid=batch.valid.tolist(),
                acquisition_errors=list(batch.acquisition_errors),
                timestamp=batch.timestamp,
                observation_revision=batch.observation_revision,
                env_ids=batch.env_ids.tolist(),
            )
            for name, batch in frame.batches.items()
        },
    )


def _window(value: ControlObservationSnapshot) -> dict[str, Any]:
    if type(value) is not ControlObservationSnapshot:
        raise TypeError("Control history must come from the public owned snapshot API.")
    return dict(
        generation=value.generation,
        legacy_time_origin=value.legacy_time_origin,
        physics_dt=value.physics_dt,
        errors=list(value.errors),
        records=[
            dict(
                frame=_frame(record.frame, kind="joint_contact_acquisition"),
                phase=record.phase,
                native_revision=record.native_revision,
                topology_revision=record.topology_revision,
                dropped_contacts=record.dropped_contacts,
            )
            for record in value.records
        ],
        samples=[_diagnostic(asdict(sample)) for sample in value.samples],
        trace=_diagnostic(deepcopy(list(value.trace))),
    )


def _measurements(value: ControlPolicyObservation) -> dict[str, Any]:
    if type(value) is not ControlPolicyObservation:
        raise TypeError(
            "Control policy facts must come from their exact public snapshot owner."
        )
    frames = {"joint_limits", "reference_qpos", "current_qpos", "clearance"}
    result = {
        item.name: deepcopy(getattr(value, item.name))
        for item in fields(value)
        if item.name not in {*frames, "hold_frames"}
    }
    if result["window"] is not None:
        result["window"] = list(result["window"])
    result["hold_frames"] = [
        _frame(frame, kind="control_policy_read") for frame in value.hold_frames
    ]
    result.update(
        {
            name: (
                None
                if getattr(value, name) is None
                else _frame(getattr(value, name), kind="control_policy_read")
            )
            for name in frames
        }
    )
    return result


class SimulationControlObservationCollector:
    """Bind one initialized bridge/episode and copy only its retained raw history.

    Construct before execution. Captures use the actual bridge's public policy
    owner, never a factory template. Completed data may be copied after native
    hooks close; it retains the original episode and reads no current state.
    """

    def __init__(
        self, environment: EmbodiedEnv, bridge: TaskProgramDemoBridge, operation_id: str
    ) -> None:
        """Bind the actual runtime instance and authentic initial episode identity.

        Args:
            environment: Initialized environment owning this bridge's simulation.
            bridge: Actual execution bridge before its first control step.
            operation_id: Explicit configured operation identifier.
        """
        from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
        from .policies import SimulationSegmentPolicyPort
        from ..invocation_policy import task_program_fingerprint

        if type(bridge) is not TaskProgramDemoBridge or bridge.clock.step_index != 0:
            raise ValueError(
                "Bind control observations to the actual fresh bridge before execution."
            )
        port = bridge.post_policy_port
        if type(port) is not SimulationSegmentPolicyPort:
            raise ValueError(
                "Control observations belong to another actual environment owner."
            )
        port.validate_environment_owner(environment.sim, environment.robot)
        episode_owner = environment.episode_context
        epochs = episode_owner.snapshot()
        if (
            len(epochs) != 1
            or not epochs[0]["initialized"]
            or type(epochs[0]["episode"]) is not int
            or epochs[0]["episode"] < 0
        ):
            raise ValueError(
                "Control observation requires one initialized native episode."
            )
        scope = port.control_operation_scope(operation_id)
        if scope["program_fingerprint"] != task_program_fingerprint(bridge.program):
            raise ValueError("Control source scope differs from this compiled program.")
        if scope["episode_id"] != str(epochs[0]["episode"]):
            raise ValueError("Control source scope differs from its actual episode.")
        port.validate_environment_owner(environment.sim, environment.robot)
        if (
            environment.episode_context is not episode_owner
            or episode_owner.snapshot() != epochs
        ):
            raise ValueError("Control episode changed during observation binding.")
        self._bridge, self._port, self._program = bridge, port, bridge.program
        self._episodes = (epochs[0]["episode"],)
        self._route = deepcopy(scope.pop("route_payload"))
        self._identity = deepcopy(scope)

    @property
    def episode_ids(self) -> tuple[int, ...]:
        """Return the captured initialized native episode IDs."""
        return self._episodes

    @property
    def identity(self) -> dict[str, Any]:
        """Return the actual instance/program/episode/native acquisition identity."""
        return deepcopy(self._identity)

    @property
    def route_payload(self) -> dict[str, Any]:
        """Return the actual owner's plain route declaration without choosing criteria."""
        return deepcopy(self._route)

    def _policy(self, selector: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
        required = {
            "segment_id",
            "segment_name",
            "source_path",
            "policy_index",
            "preset",
            "phase",
            "terminal",
        }
        if (
            not isinstance(selector, Mapping)
            or set(selector) != required
            or type(selector["policy_index"]) is not int
            or type(selector["terminal"]) is not bool
        ):
            raise ValueError(
                "Control acquisition requires a complete exact compiled policy selector."
            )
        segments = [
            segment
            for segment in self._program.iter_segments()
            if segment.segment_id == selector["segment_id"]
            and segment.name == selector["segment_name"]
            and segment.source_path == tuple(selector["source_path"])
        ]
        if len(segments) != 1:
            raise ValueError("Control acquisition has no unique compiled segment.")
        segment = segments[0]
        index = selector["policy_index"]
        if not 0 <= index < len(segment.post_policies):
            raise ValueError("Control policy index is outside its declared occurrence.")
        policy = segment.post_policies[index]
        scope = self._port.control_policy_scope(policy, segment=segment)
        if (
            any(scope.get(key) != value for key, value in self._identity.items())
            or scope["route_payload"] != self._route
            or scope["source_path"] != [*selector["source_path"], "post", index]
            or scope["policy_index"] != index
            or any(
                scope[key] != selector[key] for key in ("preset", "phase", "terminal")
            )
        ):
            raise ValueError(
                "Control policy scope differs from the bound occurrence/native owner."
            )
        return segment, policy, scope

    def capture(
        self,
        policies: Sequence[Mapping[str, Any]],
        *,
        terminal_policy: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Copy original completed policy facts and the requested retained window.

        Args:
            policies: Explicit compiled selectors, retaining caller-declared order.
            terminal_policy: Exact completed policy whose complete window is requested.

        Returns:
            Owned raw channels, receipts and immutable scope metadata. No event,
            release, clearance or final task acceptance is computed here.
        """
        from ..invocation_policy import task_program_fingerprint

        if (
            self._bridge.post_policy_port is not self._port
            or self._bridge.program is not self._program
            or task_program_fingerprint(self._program)
            != self._identity["program_fingerprint"]
        ):
            raise ValueError("The bound control bridge/program owner changed.")
        terminal_segment, terminal, scope = self._policy(terminal_policy)
        final_window = self._port.control_policy_window(
            terminal, segment=terminal_segment
        )
        records = []
        for selector in policies:
            segment, policy, selected_scope = self._policy(selector)
            facts = self._port.control_policy_measurements(policy, segment=segment)
            packet = dict(
                selector=deepcopy(dict(selector)),
                scope=selected_scope,
                measurements=_measurements(facts),
                receipt=deepcopy(
                    self._port.post_policy_metadata(policy, segment=segment)
                ),
            )
            if (
                selected_scope["family"] == "twist"
                and selected_scope["phase"] == "ready"
            ):
                packet["startup_window"] = _window(
                    self._port.control_policy_window(
                        policy, segment=segment, startup=True
                    )
                )
            records.append(packet)
        facts = self._port.control_policy_measurements(
            terminal, segment=terminal_segment
        )
        return dict(
            schema_version="simulation.control_observation/v1",
            control_identity=self.identity,
            episode_ids=list(self._episodes),
            boundary_frames=[facts.hold_end_step],
            terminal_selector=deepcopy(dict(terminal_policy)),
            policies=records,
            window=_window(final_window),
        )
