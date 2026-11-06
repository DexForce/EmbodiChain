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

"""Actual-bridge physical acquisition and independent TaskSpec run reporting."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any, TYPE_CHECKING
import yaml
from uuid import uuid4

from embodichain.gen_sim.task_spec import canonical_hash
from .bundle_spec import decode_bundle_task_spec, bundle_task_spec_payload
from embodichain.gen_sim.task_spec.evidence import (
    SupportedPlacementEvidenceBinding,
)
from embodichain.gen_sim.task_spec.condition_evidence import ConditionEvidenceBinding
from embodichain.gen_sim.task_spec.pour_evidence import PourEvidenceBinding
from embodichain.lab.task_program.integrations.evidence import HandOverEvidenceBinding
from embodichain.lab.task_program.integrations.evidence import (
    CoordinatedHoldEvidenceBinding,
)
from embodichain.lab.task_program.integrations.evidence import (
    CoordinatedTransportEvidenceBinding,
)
from embodichain.gen_sim.task_spec.placement_group_evidence import (
    PlacementEvidenceGroupBinding,
)
from embodichain.gen_sim.task_spec.condition_group_evidence import (
    ConditionEvidenceGroupBinding,
)
from embodichain.gen_sim.task_spec.control_evidence import ControlEvidenceBinding
from embodichain.lab.task_program.integrations.evidence import (
    held_observation_request,
    sampled_support_window,
    directional_observation_axes,
    validate_placement_declaration,
)
from .evidence_binding import (
    TaskEngineManipulationBinding,
    evaluate_task_engine_evidence,
    _decode_binding,
)
from .bundle_deployment import LoadedTaskProgramBundle
from .bundle_io import BundleSnapshot

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.envs.demo import DemoEpisodeResult, DemoSegmentResult
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
    from embodichain.lab.task_program.integrations.simulation.recorded_contacts import (
        SimulationRecordedContactCfg,
    )

__all__ = ["TaskEngineEvidenceReport", "TaskEngineEvidenceSession"]
_Binding = (
    SupportedPlacementEvidenceBinding
    | PlacementEvidenceGroupBinding
    | ConditionEvidenceBinding
    | ConditionEvidenceGroupBinding
    | PourEvidenceBinding
    | HandOverEvidenceBinding
    | CoordinatedHoldEvidenceBinding
    | CoordinatedTransportEvidenceBinding
    | ControlEvidenceBinding
    | TaskEngineManipulationBinding
)


def _one(values: Any, description: str) -> Any:
    items = list(values)
    if len(items) != 1:
        raise ValueError("Evidence collection requires one exact " + description + ".")
    return items[0]


@dataclass(frozen=True, slots=True)
class TaskEngineEvidenceReport:
    """Separated program outcome, original action effects and strict task judgments."""

    bundle_digest: str
    episode_ids: tuple[int, ...]
    program_success: tuple[bool, ...]
    program_completed: tuple[bool, ...]
    terminal_reasons: tuple[str, ...]
    lengths: tuple[int, ...]
    action_physical_effects: tuple[Mapping[str, Any], ...]
    strict_task_spec: tuple[Mapping[str, Any], ...]
    acquisition_errors: Mapping[str, str]
    cargo_observation: Mapping[str, Any] | None = None
    run_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self) -> None:
        for key in (
            "action_physical_effects",
            "strict_task_spec",
            "acquisition_errors",
            "cargo_observation",
        ):
            object.__setattr__(self, key, deepcopy(getattr(self, key)))

    def to_dict(self) -> dict[str, Any]:
        """Return independently owned report data without conflating success layers."""
        result = dict(
            schema_version="task_engine.physical_run_report/v1",
            bundle_digest=self.bundle_digest,
            run_id=self.run_id,
            episode_ids=list(self.episode_ids),
            program_success=list(self.program_success),
            program_completed=list(self.program_completed),
            terminal_reasons=list(self.terminal_reasons),
            lengths=list(self.lengths),
            action_physical_effects=deepcopy(list(self.action_physical_effects)),
            strict_task_spec=deepcopy(list(self.strict_task_spec)),
            acquisition_errors=dict(self.acquisition_errors),
        )
        if self.cargo_observation is not None:
            result["cargo_observation"] = deepcopy(dict(self.cargo_observation))
        return result


class TaskEngineEvidenceSession:
    """Collect only through public Lab owners, then delegate judgment to TaskSpec.

    Configure the environment through ``LoadedTaskProgramBundle.configure_environment``
    and select its bridge with ``env.create_demo_segments`` before construction.
    Bind recorded declarations before execution and pass ``capture_boundary`` as
    the standard executor's ``on_segment_completed`` callback. After execution,
    callers may project fresh final bindings against ``runtime_metadata`` and
    ``final_boundary`` before calling ``finish``. No binding is silently resigned.
    """

    def __init__(
        self,
        bundle: LoadedTaskProgramBundle,
        environment: EmbodiedEnv,
        bridge: TaskProgramDemoBridge,
        *,
        timeline_cfg: SimulationRecordedContactCfg | None = None,
        control_operation_ids: Sequence[str] = (),
    ) -> None:
        """Bind the selected initialized run and attach requested observers at step zero.

        Args:
            bundle: Validated sealed declaration bundle.
            environment: Actual initialized environment produced from that bundle.
            bridge: Actual selected bridge; no replacement runtime is constructed.
            timeline_cfg: Explicit existing-sensor declaration when sampled history is needed.
            control_operation_ids: Explicit operations whose actual scopes must be retained.
        """
        from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
        from embodichain.lab.task_program.integrations.invocation_policy import (
            task_program_fingerprint,
        )
        from embodichain.lab.task_program.integrations.simulation.observation_services import (
            SimulationObservationServices,
        )

        if (
            type(bundle) is not LoadedTaskProgramBundle
            or type(bridge) is not TaskProgramDemoBridge
        ):
            raise TypeError(
                "Evidence collection requires the actual loaded bundle and selected bridge."
            )
        if (
            environment.active_task_program_bridge is not bridge
            or bridge.clock.step_index != 0
        ):
            raise ValueError(
                "Evidence collection must bind the selected bridge before its first step."
            )
        sealed = BundleSnapshot(bundle.snapshot.identity, bundle.snapshot.files)
        identity = sealed.identity.to_dict()
        if task_program_fingerprint(bridge.program) != identity["program_fingerprint"]:
            raise ValueError("Selected bridge differs from the sealed bundle program.")
        self._program = yaml.safe_load(sealed.files["task_program/program.yaml"])
        self._spec = decode_bundle_task_spec(json.loads(sealed.files["task_spec.json"]))
        if (
            canonical_hash(bundle_task_spec_payload(self._spec))
            != identity["task_spec_digest"]
        ):
            raise ValueError("The sealed normative specification has another identity.")
        registration = bundle.deployment.integration.registration
        registration.assert_unchanged()
        if (
            canonical_hash(registration.fingerprint_payload())
            != identity["registration_fingerprint"]
        ):
            raise ValueError(
                "The supplied runtime registration differs from the sealed bundle."
            )
        ids = tuple(control_operation_ids)
        if len(set(ids)) != len(ids) or any(
            type(value) is not str or not value for value in ids
        ):
            raise ValueError("Control operation IDs must be explicit and unique.")
        self._env, self._bridge = environment, bridge
        self._run_id = str(uuid4())
        self._program_fingerprint = identity["program_fingerprint"]
        self._bundle_digest = sealed.identity.digest
        declared_timeline = registration.recorded_contacts_cfg
        if timeline_cfg is not None and (
            declared_timeline is None
            or timeline_cfg.snapshot().to_dict()
            != declared_timeline.snapshot().to_dict()
        ):
            raise ValueError(
                "Contact history must match its sealed runtime declaration; regenerate the bundle."
            )
        self._services = SimulationObservationServices(
            environment,
            bridge,
            registration,
            observations=environment.observation_manager,
        )
        try:
            self._state = self._services.state
            self._support = self._services.support
            self._episodes = self._state.episode_ids
            self._count = len(self._episodes)
            if environment.num_envs != self._count:
                raise ValueError("Environment and initialized episode rows disagree.")
            self._controls = {key: self._services.controls[key] for key in ids}
            self._timeline = self._services.recorded_contacts
            self._cargo = self._services.cargo
        except BaseException:
            self._services.close()
            raise

        self._bindings: dict[str, _Binding] = {}
        self._declarations: dict[str, Any] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._segments: list[dict[str, Any]] = []
        self._errors: dict[str, str] = {}
        self._cache: dict[str, Any] = {}
        self._closed = False
        self._finished = False

    @property
    def episode_ids(self) -> tuple[int, ...]:
        """Return the original initialized native row identities."""
        return self._episodes

    @property
    def contact_identity(self) -> dict[str, Any]:
        """Return the explicit timeline owner's identity for fresh raw binding projection."""
        if self._timeline is None:
            raise ValueError("No sampled contact collector was declared.")
        return self._timeline.identity

    def control_identity(self, operation_id: str) -> dict[str, Any]:
        """Return the actual declared operation instance's acquisition identity."""
        return self._controls[operation_id].identity

    def control_route_payload(self, operation_id: str) -> dict[str, Any]:
        """Return the original runtime owner's plain source route, without rewriting IDs."""
        return self._controls[operation_id].route_payload

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        """Return owned actual boundary records for declaration projection/reporting."""
        return deepcopy(
            dict(
                segments=self._segments,
                lengths=(
                    list(self._segments[-1]["end_steps"])
                    if self._segments
                    else [0] * self._count
                ),
            )
        )

    @property
    def final_boundary(self) -> dict[str, Any] | None:
        """Return the actual last observed boundary, never a planned future Park."""
        if not self._segments:
            return None
        segment = self._segments[-1]
        meta = segment["metadata"]
        return deepcopy(
            dict(
                segment_name=segment["name"],
                segment_id=meta["program_segment_id"],
                source_path=meta["program_segment_source_path"],
                boundary_frames=segment["end_steps"],
            )
        )

    @property
    def snapshots(self) -> dict[str, dict[str, Any]]:
        """Copy acquisition packets explicitly when raw artifact persistence is required."""
        return deepcopy(self._snapshots)

    @property
    def binding_declarations(self) -> tuple[dict[str, Any], ...]:
        """Copy every effective verified declaration, including unobserved events.

        Returns:
            Predicate-sorted complete owner serializations with their original
            binding hashes, episode/occurrence scope and acquisition identities.
            This performs no projection, acquisition or identity reconstruction
            and remains available after the session closes.
        """
        return tuple(
            self._declarations[key].to_dict() for key in sorted(self._bindings)
        )

    def _guard(self) -> None:
        from embodichain.lab.task_program.integrations.invocation_policy import (
            task_program_fingerprint,
        )

        if self._closed:
            raise RuntimeError("Evidence collection session is closed.")
        if (
            self._env.active_task_program_bridge is not self._bridge
            or task_program_fingerprint(self._bridge.program)
            != self._program_fingerprint
        ):
            raise ValueError("The selected evidence bridge/program owner changed.")
        epochs = self._env.episode_context.snapshot()
        if tuple(row["episode"] for row in epochs) != self._episodes or not all(
            row["initialized"] for row in epochs
        ):
            raise ValueError(
                "Evidence collection belongs to another initialized episode."
            )

    def _validated(self, value: _Binding | Mapping[str, Any]) -> _Binding:
        if self._spec is None:
            raise ValueError(
                "A bundle without TaskSpec cannot bind normative evidence."
            )
        declaration = _decode_binding(value)
        if isinstance(
            declaration,
            (
                HandOverEvidenceBinding,
                CoordinatedHoldEvidenceBinding,
                CoordinatedTransportEvidenceBinding,
            ),
        ):
            raise ValueError(
                "Manipulation evidence requires explicit paired acceptance criteria; regenerate the bundle."
            )
        binding = (
            declaration.runtime
            if isinstance(declaration, TaskEngineManipulationBinding)
            else declaration
        )
        if binding.evidence_format != "raw_snapshot":
            raise ValueError(
                "Live evidence sessions do not replay historical record formats."
            )
        if (
            binding.task_spec_digest != canonical_hash(self._spec.to_dict())
            or binding.program_digest != canonical_hash(self._program)
            or binding.episode_ids != self._episodes
            or binding.env_ids != tuple(range(self._count))
        ):
            raise ValueError(
                "Evidence binding differs from this sealed program/spec/episode/row scope."
            )
        if (
            getattr(binding, "program_fingerprint", self._program_fingerprint)
            != self._program_fingerprint
        ):
            raise ValueError("Evidence binding names another compiled program.")
        for member in (
            binding.members
            if isinstance(
                binding, (ConditionEvidenceGroupBinding, PlacementEvidenceGroupBinding)
            )
            else (binding,)
        ):
            self._validate_native_binding_owner(member)
            if isinstance(member, SupportedPlacementEvidenceBinding):
                validate_placement_declaration(member.to_dict())
        # Public TaskSpec validation checks predicate/family/goal-versus-milestone
        # association without reading a simulator or inventing missing evidence.
        evaluate_task_engine_evidence(
            self._spec, (declaration,), {}, program=self._program, num_envs=self._count
        )
        self._declarations[binding.predicate_id] = declaration
        return binding

    def _validate_native_binding_owner(self, binding: _Binding) -> None:
        if isinstance(binding, ControlEvidenceBinding):
            op = binding.operation["route_payload"]["operation_id"]
            if (
                op not in self._controls
                or binding.control_identity != self._controls[op].identity
            ):
                raise ValueError(
                    "Control evidence was not bound to this actual runtime instance before execution."
                )
        native = getattr(binding, "sampled_contact_identity", None)
        if native is not None and (
            self._timeline is None or native != self._timeline.identity
        ):
            raise ValueError(
                "Sampled evidence differs from its declared actual timeline owner."
            )

    def bind(self, bindings: Sequence[_Binding | Mapping[str, Any]]) -> None:
        """Install explicit declarations before any boundary can be missed.

        Args:
            bindings: Sealed evidence declarations, including separate manipulation acceptance criteria.
        """
        self._guard()
        if self._bridge.clock.step_index or self._segments:
            raise ValueError("Bind evidence declarations before execution begins.")
        selected = [self._validated(value) for value in bindings]
        if len({item.predicate_id for item in selected}) != len(selected):
            raise ValueError("Evidence bindings require unique predicate identities.")
        self._bindings = {item.predicate_id: item for item in selected}

    def record_unavailable(self, predicate_id: str, reason: str) -> None:
        """Keep an explicit declaration/acquisition gap without supplying a verdict.

        Args:
            predicate_id: Existing goal or milestone predicate in this TaskSpec.
            reason: Concrete unavailable proof or acquisition channel.

        This records diagnostics only. It cannot supply measurements, replace a
        binding, or turn an unsupported predicate into physical success/failure.
        """
        self._guard()
        if self._spec is None or predicate_id not in {
            item.id
            for item in (
                *self._spec.goal,
                *(
                    item
                    for milestone in self._spec.milestones
                    for item in milestone.achieve
                ),
            )
        }:
            raise ValueError(
                "Unavailable evidence requires an existing normative predicate."
            )
        if type(reason) is not str or not reason.strip():
            raise ValueError("Unavailable evidence requires its concrete reason.")
        self._errors[predicate_id] = reason

    def _predicate(self, binding: _Binding) -> Any:
        return _one(
            (
                item
                for item in (
                    *self._spec.goal,
                    *(
                        item
                        for milestone in self._spec.milestones
                        for item in milestone.achieve
                    ),
                )
                if item.id == binding.predicate_id
            ),
            "predicate declaration",
        )

    def _boundary(
        self, binding: _Binding, runtime: Mapping[str, Any]
    ) -> dict[str, Any]:
        if isinstance(
            binding, (HandOverEvidenceBinding, CoordinatedHoldEvidenceBinding)
        ):
            key = (
                "transfer"
                if isinstance(binding, HandOverEvidenceBinding)
                and self._predicate(binding).name == "handover_complete"
                else "boundary"
            )
            descriptor = binding.operation[key]
            return _one(
                (
                    item
                    for item in runtime["segments"]
                    if item["metadata"]["program_segment_id"]
                    == descriptor["segment_id"]
                    and item["metadata"]["program_segment_source_path"]
                    == descriptor["source_path"]
                ),
                "held execution boundary",
            )
        if isinstance(binding, PourEvidenceBinding):
            return _one(
                (
                    item
                    for item in runtime["segments"]
                    if item["metadata"]["program_segment_id"]
                    == binding.boundary_segment_id
                ),
                "Pour execution boundary",
            )
        name = (
            binding.operation["terminal_segment_id"]
            if isinstance(binding, ControlEvidenceBinding)
            else binding.boundary_segment_id
        )
        path = (
            binding.operation["terminal_source_path"]
            if isinstance(binding, ControlEvidenceBinding)
            else binding.boundary_source_path
        )
        return _one(
            (
                item
                for item in runtime["segments"]
                if item["name"] == name
                and item["metadata"]["program_segment_source_path"] == list(path)
            ),
            "declared execution boundary",
        )

    def _validate_boundary(self, segment: Mapping[str, Any]) -> None:
        meta = segment["metadata"]
        declared = _one(
            (
                item
                for item in self._bridge.program.iter_segments()
                if item.segment_id == meta.get("program_segment_id")
            ),
            "selected compiled segment",
        )
        if (
            segment["name"] != declared.name
            or meta.get("task_program_id") != self._bridge.program.program_id
            or meta.get("program_segment_source_path") != list(declared.source_path)
        ):
            raise ValueError(
                "Executor boundary differs from the selected compiled program."
            )
        for key in ("start_steps", "end_steps"):
            values = segment.get(key)
            if (
                not isinstance(values, list)
                or len(values) != self._count
                or any(type(value) is not int or value < 0 for value in values)
            ):
                raise ValueError(
                    "Executor boundary lacks exact native row/frame identities."
                )
        if any(
            start > end
            for start, end in zip(
                segment["start_steps"], segment["end_steps"], strict=True
            )
        ):
            raise ValueError("Executor boundary has reversed native frame intervals.")

    def _once(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        key = canonical_hash(dict(method=method.__name__, args=args, kwargs=kwargs))
        if key not in self._cache:
            try:
                self._cache[key] = method(*args, **kwargs)
            except Exception as error:
                self._cache[key] = error
        value = self._cache[key]
        if isinstance(value, Exception):
            raise value
        return value

    def _capture(self, binding: _Binding, runtime: Mapping[str, Any]) -> dict[str, Any]:
        self._guard()
        common = dict(
            segments=runtime["segments"],
            lengths=runtime["lengths"],
            episode_ids=list(self._episodes),
            run_id=self._run_id,
        )
        if (
            isinstance(binding, ConditionEvidenceBinding)
            and binding.observation_kind == "final"
        ):
            packet = (
                self._once(self._state.capture_current_rigid_pose, binding.entity_id)
                if binding.evidence_kind == "orientation"
                else self._once(
                    self._state.capture_current_joint_position,
                    binding.entity_id,
                    binding.condition["joint"],
                )
            )
            return {**common, **packet}
        boundary = self._boundary(binding, runtime)
        frames = boundary["end_steps"]
        common["boundary_frames"] = list(frames)
        recorded = binding.observation_kind == "recorded"
        if isinstance(binding, SupportedPlacementEvidenceBinding):
            if binding.sampled_contact_identity is not None:
                starts, ends = sampled_support_window(binding.to_dict(), frames)
                packet = self._once(
                    self._timeline.capture,
                    binding.object_id,
                    binding.reference_id,
                    start_frame=starts[0],
                    end_frame=ends[0],
                )
                return {**common, "recorded_contacts": packet}
            if binding.planar_geometry is not None:
                packet = self._once(
                    self._state.capture_planar_placement,
                    binding.object_id,
                    binding.reference_id,
                    frames,
                    observation_kind=binding.observation_kind,
                    exact_silhouette=binding.planar_geometry["exact_silhouette"],
                )
            elif binding.directional_geometry is not None:
                source = binding.directional_geometry
                packet = self._once(
                    self._state.capture_directional_placement,
                    binding.object_id,
                    binding.reference_id,
                    source["table_id"],
                    frames,
                    axes=directional_observation_axes(binding.to_dict()),
                    observation_kind=binding.observation_kind,
                    preset_id=binding.stability_preset,
                    reference_part=source["reference_part"],
                    reference_link_id=source["reference_link_id"],
                    segment_name=binding.boundary_segment_id,
                    source_path=binding.boundary_source_path,
                )
            elif recorded:
                packet = self._once(
                    self._support.capture_recorded,
                    binding.stability_preset,
                    segment_id=binding.support_segment_id,
                    source_path=binding.support_source_path,
                    frame_indices=frames,
                )
            else:
                packet = self._once(
                    self._support.capture_current, binding.stability_preset
                )
            return {**common, **packet}
        if isinstance(binding, ConditionEvidenceBinding):
            packet = (
                self._once(
                    self._state.capture_recorded_rigid_pose, binding.entity_id, frames
                )
                if binding.evidence_kind == "orientation"
                else self._once(
                    self._state.capture_recorded_joint_policy,
                    binding.entity_id,
                    preset_id=binding.stability_preset,
                    segment_id=binding.policy_segment_id,
                    source_path=binding.policy_source_path,
                    frame_indices=frames,
                )
            )
            return {**common, **packet}
        if isinstance(binding, PourEvidenceBinding):
            segment = _one(
                (
                    item
                    for item in runtime["segments"]
                    if item["metadata"]["program_segment_id"] == binding.pour_segment_id
                ),
                "Pour operation interval",
            )
            packet = self._once(
                self._state.capture_recorded_pour_geometry,
                binding.object_id,
                binding.reference_id,
                start_frames=segment["start_steps"],
                end_frames=segment["end_steps"],
            )
            return {**common, **packet}
        if isinstance(
            binding, (HandOverEvidenceBinding, CoordinatedHoldEvidenceBinding)
        ):
            request = held_observation_request(binding, runtime, program=self._program)
            common.update(
                boundary_frames=request["boundary_frames"],
                program_fingerprint=self._program_fingerprint,
            )
            if request["capture_pose"]:
                common["held_poses"] = self._once(
                    self._state.capture_held_poses,
                    request["object_id"],
                    request["motion_parts"],
                    request["boundary_frames"],
                    observation_kind=request["pose_observation_kind"],
                )
            grips = {
                item["resource_id"]: self._once(
                    self._state.capture_current_grip,
                    request["object_id"],
                    item["control_part"],
                    observation_revision=item["observation_revision"],
                    frame_indices=request["boundary_frames"],
                )
                for item in request["grips"]
            }
            if isinstance(binding, HandOverEvidenceBinding):
                if grips:
                    common["fresh_grip"] = grips[
                        binding.operation["destination_resource"]
                    ]
            else:
                common["fresh_grips"] = {
                    role: grips[resource]
                    for role, resource in binding.operation["resources"].items()
                }
            return common
        if isinstance(binding, CoordinatedTransportEvidenceBinding):
            return {
                **common,
                "recorded_contacts": self._once(
                    self._timeline.capture_timeline,
                    binding.object_id,
                    binding.reference_id,
                    boundary_frame=frames[0],
                ),
            }
        if isinstance(binding, ControlEvidenceBinding):
            collector = self._controls[
                binding.operation["route_payload"]["operation_id"]
            ]
            packet = collector.capture(
                binding.policy_occurrences,
                terminal_policy=binding.policy_occurrences[-1],
            )
            return {**common, "control_acquisition": packet}
        raise ValueError("Unsupported physical evidence binding.")

    def _packet_or_unavailable(
        self, binding: _Binding, runtime: Mapping[str, Any]
    ) -> dict[str, Any]:
        try:
            packet = self._capture(binding, runtime)
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            self._errors[binding.predicate_id] = reason
            packet = dict(
                unavailable=reason,
                episode_ids=list(self._episodes),
                run_id=self._run_id,
            )
            try:
                packet["boundary_frames"] = list(
                    self._boundary(binding, runtime)["end_steps"]
                )
            except (ValueError, KeyError, TypeError):
                pass

        return packet

    @staticmethod
    def _group_member_runtime(
        binding: ConditionEvidenceBinding | SupportedPlacementEvidenceBinding,
        runtime: Mapping[str, Any],
    ) -> dict[str, Any]:
        # Each recorded state needs its own exact execution prefix, not another
        # copy of every preceding object's runtime. Keep group storage linear in
        # the declared member prefixes. Current goals retain the actual terminal
        # occurrence, independent of whether that member was executed.
        names = {value["segment_id"] for value in binding.required_execution}
        if (
            isinstance(binding, SupportedPlacementEvidenceBinding)
            and binding.observation_kind == "final"
        ):
            names.update((binding.release_segment_id, binding.boundary_segment_id))
            if binding.evidence_scope == "native_supported_placement":
                names.add(binding.support_segment_id)
        segments = (
            [value for value in runtime["segments"] if value["name"] in names]
            if binding.observation_kind == "recorded"
            or isinstance(binding, SupportedPlacementEvidenceBinding)
            else runtime["segments"][-1:]
        )
        return dict(segments=segments, lengths=runtime["lengths"])

    def _group_packet(
        self,
        binding: ConditionEvidenceGroupBinding | PlacementEvidenceGroupBinding,
        members: Mapping[str, Any],
    ) -> dict[str, Any]:
        packet = dict(
            schema_version=(
                "task_spec.placement_group_observation/v1"
                if isinstance(binding, PlacementEvidenceGroupBinding)
                else "task_spec.condition_group_observation/v1"
            ),
            binding_hash=binding.binding_hash,
            episode_ids=list(self._episodes),
            run_id=self._run_id,
            members=dict(members),
        )
        frames = [
            members.get(value.binding_hash, {}).get("boundary_frames")
            for value in binding.members
        ]
        if binding.observation_kind == "recorded" and all(
            isinstance(row, list)
            and len(row) == self._count
            and all(type(value) is int and value > 0 for value in row)
            for row in frames
        ):
            packet["boundary_frames"] = [
                max(row[i] for row in frames) for i in range(self._count)
            ]
        return packet

    def _capture_or_unavailable(
        self, binding: _Binding, runtime: Mapping[str, Any]
    ) -> None:
        if isinstance(
            binding, (ConditionEvidenceGroupBinding, PlacementEvidenceGroupBinding)
        ):
            members = {
                value.binding_hash: self._packet_or_unavailable(
                    value, self._group_member_runtime(value, runtime)
                )
                for value in binding.members
            }
            packet = self._group_packet(binding, members)
        else:
            packet = self._packet_or_unavailable(binding, runtime)
        self._snapshots[binding.predicate_id] = packet

    def capture_boundary(self, result: DemoSegmentResult) -> None:
        """Observe a completed executor boundary before the next segment begins.

        Args:
            result: Original result emitted after ordinary recording, including failures.
        """
        from embodichain.lab.gym.envs.demo import DemoSegmentResult

        if type(result) is not DemoSegmentResult:
            raise TypeError(
                "Evidence boundaries require the executor's actual segment result."
            )
        if self._finished or self._closed:
            raise RuntimeError("Evidence session is no longer accepting boundaries.")
        self._guard()
        boundary_metadata = result.to_metadata()
        self._validate_boundary(boundary_metadata)
        self._segments.append(boundary_metadata)
        self._cache = {}
        runtime = self.runtime_metadata
        for binding in self._bindings.values():
            if binding.observation_kind != "recorded":
                continue
            if isinstance(
                binding, (ConditionEvidenceGroupBinding, PlacementEvidenceGroupBinding)
            ):
                packets = dict(
                    self._snapshots.get(binding.predicate_id, {}).get("members", {})
                )
                for member in binding.members:
                    try:
                        boundary = self._boundary(member, runtime)
                    except (ValueError, KeyError):
                        continue
                    if boundary != runtime["segments"][-1]:
                        continue
                    if member.binding_hash in packets:
                        reason = "The selected compiled member boundary occurred more than once."
                        packets[member.binding_hash] = dict(
                            unavailable=reason,
                            episode_ids=list(self._episodes),
                            run_id=self._run_id,
                        )
                        self._errors[binding.predicate_id] = reason
                    else:
                        packets[member.binding_hash] = self._packet_or_unavailable(
                            member, self._group_member_runtime(member, runtime)
                        )
                if packets:
                    self._snapshots[binding.predicate_id] = self._group_packet(
                        binding, packets
                    )
                continue
            try:
                boundary = self._boundary(binding, runtime)
            except (ValueError, KeyError):
                continue
            if boundary != runtime["segments"][-1]:
                continue
            if binding.predicate_id in self._snapshots:
                self._snapshots[binding.predicate_id] = dict(
                    unavailable="The selected compiled boundary occurred more than once.",
                    episode_ids=list(self._episodes),
                )
                self._errors[binding.predicate_id] = self._snapshots[
                    binding.predicate_id
                ]["unavailable"]
            else:
                self._capture_or_unavailable(binding, runtime)
        self._cache = {}

    def finish(
        self,
        result: DemoEpisodeResult,
        *,
        final_bindings: Sequence[_Binding | Mapping[str, Any]] | None = None,
    ) -> TaskEngineEvidenceReport:
        """Capture actual final state, qualify all packets and close owned observers.

        Args:
            result: Actual executor result for this selected bridge/episode.
            final_bindings: Optional freshly projected final declarations using the
                actual terminal boundary. Existing bindings are never mutated or resigned.

        Returns:
            Separated program, action-effect and strict normative outcomes.
        """
        from embodichain.lab.gym.envs.demo import DemoEpisodeResult

        if type(result) is not DemoEpisodeResult or self._finished:
            raise ValueError(
                "Evidence finish requires one actual unreported episode result."
            )
        try:
            self._guard()
            metadata = result.to_metadata()
            segments = metadata["segments"]
            if self._segments and self._segments != segments[: len(self._segments)]:
                raise ValueError(
                    "Final runtime result differs from observed executor boundaries."
                )
            if len(result.success) != self._count or len(result.lengths) != self._count:
                raise ValueError("Final runtime result has another native row count.")
            for segment in segments:
                self._validate_boundary(segment)
            if segments and list(result.lengths) != segments[-1]["end_steps"]:
                raise ValueError(
                    "Final runtime lengths differ from their actual last boundary."
                )
            self._segments = segments
            if final_bindings is not None:
                selected = [self._validated(value) for value in final_bindings]
                if any(item.observation_kind != "final" for item in selected) or len(
                    {item.predicate_id for item in selected}
                ) != len(selected):
                    raise ValueError(
                        "Fresh final bindings must have unique final predicate identities."
                    )
                retained = {
                    key: value
                    for key, value in self._bindings.items()
                    if value.observation_kind == "recorded"
                }
                if set(retained) & {item.predicate_id for item in selected}:
                    raise ValueError(
                        "Final bindings cannot replace recorded predicates."
                    )
                self._bindings = {
                    **retained,
                    **{item.predicate_id: item for item in selected},
                }
            runtime = dict(segments=segments, lengths=list(result.lengths))
            self._cache = {}
            for binding in self._bindings.values():
                if binding.observation_kind == "final":
                    self._capture_or_unavailable(binding, runtime)
                elif binding.predicate_id not in self._snapshots:
                    reason = "The exact recorded boundary was not observed by this collection session."
                    self._snapshots[binding.predicate_id] = dict(
                        unavailable=reason,
                        episode_ids=list(self._episodes),
                        run_id=self._run_id,
                    )
                    self._errors[binding.predicate_id] = reason
            if self._spec is None:
                reports = tuple(
                    dict(
                        status="unavailable",
                        certificate_ready=False,
                        reason="No normative TaskSpec was declared for this bundle.",
                        init=[],
                        goal=[],
                        invariants=[],
                        temporal=[],
                        milestones=[],
                    )
                    for _ in self._episodes
                )
            else:
                reports = tuple(
                    report.to_dict()
                    for report in evaluate_task_engine_evidence(
                        self._spec,
                        tuple(self._declarations[key] for key in self._bindings),
                        self._snapshots,
                        program=self._program,
                        num_envs=self._count,
                    )
                )
            effects = tuple(
                dict(
                    segment_name=segment["name"],
                    segment_id=segment["metadata"].get("program_segment_id"),
                    source_path=segment["metadata"].get("program_segment_source_path"),
                    invocation_id=call.get("invocation_id"),
                    invocation_revision=call.get("invocation_revision"),
                    status=call.get("status"),
                    effects=deepcopy(call.get("effects", [])),
                )
                for segment in segments
                for call in segment["metadata"].get("runtime", {}).get("calls", [])
            )
            cargo = None
            if self._cargo is not None:
                invocation = next(
                    (
                        value["invocation_id"]
                        for value in reversed(effects)
                        if type(value.get("invocation_id")) is str
                    ),
                    None,
                )
                try:
                    if invocation is None:
                        raise ValueError(
                            "No actual terminal invocation is available for cargo observation."
                        )
                    cargo = self._cargo.capture_report(invocation_id=invocation)
                except (
                    RuntimeError,
                    ValueError,
                    KeyError,
                    TypeError,
                    AttributeError,
                ) as error:
                    reason = str(error) or type(error).__name__
                    self._errors["cargo_envelope"] = reason
                    cargo = dict(
                        status="unavailable",
                        reason=reason,
                        episode_ids=list(self._episodes),
                    )
            self._finished = True
            return TaskEngineEvidenceReport(
                self._bundle_digest,
                self._episodes,
                tuple(result.success),
                tuple(result.completed_by_env or (result.completed,) * self._count),
                tuple(
                    result.terminal_reasons or (result.terminal_reason,) * self._count
                ),
                tuple(result.lengths),
                effects,
                reports,
                dict(self._errors),
                cargo,
                run_id=self._run_id,
            )
        finally:
            self._cache = {}
            self.close()

    def close(self) -> None:
        """Detach only observers owned by this session, never the borrowed policy port."""
        if not self._closed:
            self._services.close()
            self._closed = True

    def __enter__(self) -> TaskEngineEvidenceSession:
        """Return this already-bound collection session."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Release owned observers if execution exits before final capture."""
        self.close()
