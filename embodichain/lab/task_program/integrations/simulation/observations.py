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

"""Owned current/recorded physical support snapshots for a bound Gym execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from typing import Any, TYPE_CHECKING

from .policies import SimulationSegmentPolicyPort
from .stability import SimulationStabilityConditionCfg

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
    from ..catalog import SimulationTaskProgramRegistration

__all__ = [
    "SimulationSupportObservationCollector",
    "SimulationStateObservationCollector",
]


class _SimulationObservationCollector:
    """Borrow one execution and its optional canonical observation history."""

    def __init__(
        self,
        environment: EmbodiedEnv,
        bridge: TaskProgramDemoBridge,
        registration: SimulationTaskProgramRegistration,
        *,
        observations: ObservationManager | None = None,
    ) -> None:
        from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
        from ..catalog import SimulationTaskProgramRegistration, _canonical_value

        if (
            type(bridge) is not TaskProgramDemoBridge
            or type(registration) is not SimulationTaskProgramRegistration
        ):
            raise TypeError(
                "Support collection requires an exact bridge and sealed registration."
            )
        if bridge.clock.step_index != 0 or bridge.program_completed:
            raise ValueError(
                "Bind physical observation collection before the first execution step."
            )
        port = bridge.post_policy_port
        if type(port) is not SimulationSegmentPolicyPort:
            raise TypeError("This bridge has no supported simulation policy session.")
        if (
            port._simulation is not environment.sim
            or port._robot is not environment.robot
        ):
            raise ValueError(
                "The bridge policy session belongs to another environment."
            )
        registration.assert_unchanged()
        selection = bridge.program.integration
        if (
            selection.scene_registry != registration.catalog.scene_registry_id
            or selection.robot_profile != registration.catalog.robot_profile_id
        ):
            raise ValueError(
                "The bridge program has another registered integration selection."
            )
        if _canonical_value(port._scene_binding) != _canonical_value(
            registration.scene_binding
        ):
            raise ValueError(
                "The bridge scene binding differs from the selected registration."
            )
        if set(port._settle_presets) != set(registration.settle_presets):
            raise ValueError(
                "The bridge policy table has another complete preset selection."
            )
        for name, cfg in registration.settle_presets.items():
            if name not in port._settle_presets or asdict(cfg) != asdict(
                port._settle_presets[name]
            ):
                raise ValueError(
                    "The bridge policy table differs from the selected registration."
                )
        identities = environment.episode_context.snapshot()
        if not identities or not all(
            row["initialized"] and type(row["episode"]) is int and row["episode"] >= 0
            for row in identities
        ):
            raise ValueError(
                "Physical support collection requires initialized episode rows."
            )
        if observations is not None:
            if not observations.has_history:
                raise ValueError(
                    "Historical collection requires enabled observation history."
                )
            observations.validate_history_owner(environment.episode_context)
        self._env, self._bridge, self._port = environment, bridge, port
        self._registration = registration
        self._observations = observations
        self._episodes = tuple(row["episode"] for row in identities)
        self._bindings = {
            item.entity_id: item for item in registration.scene_binding.rigid_objects
        }

    @property
    def episode_ids(self) -> tuple[int, ...]:
        """Return this collector's pre-execution EpisodeContext identities."""
        return self._episodes

    def _check_episode(self) -> None:
        self._port._require_open()
        if getattr(self._env, "_closed", False):
            raise RuntimeError(
                "Physical observation collection is unavailable after close."
            )
        current = self._env.episode_context.snapshot()
        if tuple(row["episode"] for row in current) != self._episodes or not all(
            row["initialized"] for row in current
        ):
            raise ValueError("Physical support observations belong to another episode.")
        self._registration.assert_unchanged()


class SimulationSupportObservationCollector(_SimulationObservationCollector):
    """Read physical support snapshots from one explicit environment/bridge pair.

    Construct before the first bridge control step, using the actual bridge rather
    than an adapter template port. This object borrows the existing policy
    session, registry and sensors. It creates no native resources, steps no
    physics and computes no TaskSpec predicate or release decision.

    Args:
        environment: Initialized recording environment for this execution.
        bridge: Exact bridge whose clock and policy session drive this run.
        registration: Sealed declaration used to create the runtime.
        observations: Observation manager owning the source schema and optional
            numeric history. Historical capture requires this dependency.
    """

    def _condition(self, preset_id: str) -> SimulationStabilityConditionCfg:
        cfg = self._registration.settle_presets[preset_id]
        if (
            type(cfg) is not SimulationStabilityConditionCfg
            or cfg.support_sensor is None
        ):
            raise ValueError("The selected preset has no declared support observation.")
        return cfg

    def capture_current(self, preset_id: str) -> dict[str, Any]:
        """Read the current pair through this session's existing sensor observer.

        Args:
            preset_id: Registered condition selecting the existing pair and
                support scope. Current acquisition uses the observer's native
                rest-envelope COM tolerance; a wait-policy override is confined
                to that policy's recorded measurement.

        Returns:
            Owned native metadata, validity/error channel, observed poses and
            episode IDs. The acquisition timestamp's clock domain is explicit.
            Unavailable acquisition never falls back to a geometry-only result.

        Raises:
            ValueError: The episode, declaration or runtime clock owner differs.
            RuntimeError: The environment or runtime resources are closed.
        """
        self._check_episode()
        cfg = self._condition(preset_id)
        result = self._port.capture_support_observation(
            preset_id, clock=self._bridge.clock
        )
        self._check_episode()
        channel = result.support
        native = deepcopy(dict(result.metadata))
        geometry = {}
        if (
            cfg.kind == "supported_region"
            and "object_pose" in native
            and "reference_pose" in native
        ):
            geometry = self._port.measure_support_geometry(
                preset_id,
                native_support=native,
                object_pose=native["object_pose"],
                reference_pose=native["reference_pose"],
            )
        self._check_episode()
        return {
            **geometry,
            "native_support": native,
            "boundary_poses": {
                name: deepcopy(native[key])
                for name, key in (
                    (cfg.entity, "object_pose"),
                    (cfg.reference, "reference_pose"),
                )
                if key in native
            },
            "support_acquisition": {
                "evidence_id": channel.evidence_id,
                "values": channel.values.tolist(),
                "valid": channel.valid.tolist(),
                "acquisition_errors": list(channel.acquisition_errors),
                "timestamp": channel.timestamp,
                "env_ids": channel.env_ids.tolist(),
                "observation_revision": channel.observation_revision,
                "clock_domain": (
                    "environment_step"
                    if cfg.support_contract_revision == 1
                    else "simulation_time"
                ),
            },
            "episode_ids": list(self._episodes),
        }

    def capture_recorded(
        self,
        preset_id: str,
        *,
        segment_id: str,
        source_path: Sequence[str | int],
        frame_indices: Sequence[int],
    ) -> dict[str, Any]:
        """Copy historical policy metadata and its exact recorded boundary poses.

        Args:
            preset_id: Registered support condition applied by the segment.
            segment_id: Exact compiled segment name.
            source_path: Exact compiled source path; ambiguous occurrences fail.
            frame_indices: Completed end-step slot for every environment row.

        Returns:
            Owned recorded support/acquisition metadata and seeded boundary poses.
            This path never samples current contacts or reads live body poses.

        Raises:
            ValueError: The historical selection or current episode is unqualified.
            RuntimeError: Recording or the requested policy measurement is absent.
        """
        self._check_episode()
        frames = tuple(frame_indices)
        if len(frames) != len(self._episodes) or any(
            type(frame) is not int or frame <= 0 for frame in frames
        ):
            raise ValueError(
                "Recorded support requires one positive completed boundary per row."
            )
        if self._observations is None:
            raise RuntimeError("Observation history is unavailable.")
        cfg = self._condition(preset_id)
        segments = [
            segment
            for segment in self._bridge.program.iter_segments()
            if segment.name == segment_id and segment.source_path == tuple(source_path)
        ]
        if len(segments) != 1:
            raise ValueError(
                "Recorded support requires one exact compiled segment occurrence."
            )
        segment = segments[0]
        policies = [
            policy for policy in segment.post_policies if policy.cfg.preset == preset_id
        ]
        if len(policies) != 1:
            raise ValueError(
                "Recorded support requires one declared post-policy occurrence."
            )
        try:
            metadata = self._port.post_policy_metadata(policies[0], segment=segment)
            measurements = metadata["measurements"]
            native = deepcopy(measurements["native_support"])
            acquisition = deepcopy(measurements["support_acquisition"])
        except (KeyError, RuntimeError) as error:
            raise RuntimeError(
                "Recorded native support/acquisition measurement is unavailable."
            ) from error
        from embodichain.compute.geometry.rigid import validate_rigid_pose_batch
        from embodichain.utils.math import xyz_quat_to_4x4_matrix

        fields = {}
        for name in (cfg.entity, cfg.reference):
            uid = self._bindings[name].simulation_uid
            candidates = [
                field
                for field, descriptor in self._observations.schema.items()
                if descriptor.entity_uid == uid
                and descriptor.quantity == "pose"
                and descriptor.coordinate_frame == "arena"
                and descriptor.representation == "xyz_xyzw"
                and descriptor.shape == (7,)
                and descriptor.units == ("m", "unitless")
            ]
            if not candidates:
                raise RuntimeError(
                    f"Recorded raw arena pose for {uid!r} is unavailable."
                )
            # Described aliases share their raw source; names are consumer projections.
            fields[name] = candidates[0]
        snapshot = self._observations.read_history(
            frame_indices,
            fields=tuple(dict.fromkeys(fields.values())),
            episode_ids=self._episodes,
        )
        poses = {}
        for name, field in fields.items():
            if not bool(snapshot.valid[field].all()):
                raise RuntimeError(
                    f"Recorded raw pose acquisition for {name!r} is unavailable: {snapshot.acquisition_errors[field]}."
                )
            matrices = xyz_quat_to_4x4_matrix(snapshot.values[field])
            validate_rigid_pose_batch(matrices, len(snapshot.env_ids))
            poses[name] = matrices.tolist()
        self._check_episode()
        acquisition["clock_domain"] = (
            "environment_step"
            if cfg.support_contract_revision == 1
            else "simulation_time"
        )
        geometry = {}
        if cfg.kind == "supported_region":
            geometry = self._port.measure_support_geometry(
                preset_id,
                native_support=native,
                object_pose=poses[cfg.entity],
                reference_pose=poses[cfg.reference],
            )
        self._check_episode()
        return {
            **geometry,
            "native_support": native,
            "support_acquisition": acquisition,
            "boundary_poses": poses,
            "boundary_frames": list(frame_indices),
            "episode_ids": list(self._episodes),
        }


class SimulationStateObservationCollector(_SimulationObservationCollector):
    """Capture declared rigid poses and articulated joint positions as raw facts.

    The collector uses the same pre-execution bridge/episode binding as support
    collection. It never creates an engine, changes a target or computes task
    acceptance. Current reads reject stale native owners and session clocks;
    historical poses use only the existing exact seeded recording slots.

    Args:
        environment: Initialized environment whose native owners are observed.
        bridge: Exact pre-execution bridge and its observation policy session.
        registration: Sealed scene, profile, and policy declarations.
        observations: Canonical observation manager for historical raw values.
    """

    def __init__(
        self,
        environment: EmbodiedEnv,
        bridge: TaskProgramDemoBridge,
        registration: SimulationTaskProgramRegistration,
        *,
        observations: ObservationManager | None = None,
    ) -> None:
        super().__init__(environment, bridge, registration, observations=observations)
        self._rigid_owners = {
            binding.entity_id: (
                binding.simulation_uid,
                environment.sim.get_rigid_object(binding.simulation_uid),
            )
            for binding in registration.scene_binding.rigid_objects
        }
        self._joint_owners = {
            binding.entity_id: (
                binding.simulation_uid,
                environment.sim.get_articulation(binding.simulation_uid),
            )
            for binding in registration.scene_binding.articulations
        }
        self._capture_revision = 0

    def _native(self, entity_id: str, *, articulated: bool) -> tuple[str, Any]:
        bindings = self._joint_owners if articulated else self._rigid_owners
        uid, native = bindings[entity_id]
        lookup = (
            self._env.sim.get_articulation
            if articulated
            else self._env.sim.get_rigid_object
        )
        if (
            native is None
            or lookup(uid) is not native
            or getattr(native, "uid", None) != uid
        ):
            raise ValueError("Current state belongs to another native object owner.")
        return uid, native

    def _packet(
        self,
        entity_id: str,
        native_uid: str,
        values: Any,
        *,
        kind: str,
        observation_kind: str,
        joint: str | None = None,
        frame_indices: Sequence[int] | None = None,
        source_valid: Any = None,
        source_errors: Sequence[str | None] | None = None,
    ) -> dict[str, Any]:
        import torch
        from embodichain.lab.sim.evidence import PhysicalEvidenceBatch
        from embodichain.compute.geometry.rigid import validate_rigid_pose_batch

        if (
            not isinstance(values, torch.Tensor)
            or values.ndim == 0
            or values.shape[0] != len(self._episodes)
            or not values.is_floating_point()
        ):
            raise ValueError(
                "Physical state must contain the declared floating environment rows."
            )
        values = values.detach().cpu()
        valid, errors = [], []
        for row in range(len(self._episodes)):
            if source_valid is not None and not bool(source_valid[row]):
                valid.append(False)
                errors.append(
                    source_errors[row]
                    if source_errors is not None and source_errors[row]
                    else "Recorded raw pose acquisition is unavailable."
                )
                continue
            try:
                if kind == "rigid_pose":
                    validate_rigid_pose_batch(values[row : row + 1], 1)
                elif values.ndim != 1 or not torch.isfinite(values[row]):
                    raise ValueError(
                        "Joint position observation is nonfinite or malformed."
                    )
                valid.append(True)
                errors.append(None)
            except (ValueError, TypeError, RuntimeError) as error:
                valid.append(False)
                errors.append(str(error) or type(error).__name__)
        self._capture_revision += 1
        batch = PhysicalEvidenceBatch(
            "rigid.pose" if kind == "rigid_pose" else "articulation.joint_position",
            values,
            torch.tensor(valid, dtype=torch.bool, device="cpu"),
            tuple(errors),
            float(self._env.sim.simulation_time),
            torch.arange(len(self._episodes), device="cpu"),
            self._capture_revision,
        )
        self._check_episode()
        result = dict(
            schema_version="simulation.state_observation/v1",
            entity_id=entity_id,
            native_uid=native_uid,
            observation_kind=observation_kind,
            episode_ids=list(self._episodes),
            state_acquisition=dict(
                evidence_id=batch.evidence_id,
                value_dtype=str(batch.values.dtype),
                values=[
                    value.tolist() if ok else None
                    for value, ok in zip(batch.values, valid, strict=True)
                ],
                valid=valid,
                acquisition_errors=errors,
                timestamp=batch.timestamp,
                env_ids=batch.env_ids.tolist(),
                observation_revision=batch.observation_revision,
                clock_domain="simulation_time",
                revision_kind="collector_acquisition",
            ),
        )
        if joint is not None:
            result["joint"] = joint
        if frame_indices is not None:
            result["boundary_frames"] = list(frame_indices)
        return result

    def capture_current_rigid_pose(self, entity_id: str) -> dict[str, Any]:
        """Read a canonical rigid entity from this bridge's current native owner.

        Args:
            entity_id: Exact semantic rigid-object identity in the registration.

        Returns:
            Owned pose/acquisition data with per-row validity and episode identity.
        """
        self._check_episode()
        self._port.validate_observation_scope(self._bridge.clock)
        uid, native = self._native(entity_id, articulated=False)
        values = native.get_local_pose(to_matrix=True)
        result = self._packet(
            entity_id, uid, values, kind="rigid_pose", observation_kind="final"
        )
        self._port.validate_observation_scope(self._bridge.clock)
        return result

    def capture_current_joint_position(
        self, entity_id: str, joint: str
    ) -> dict[str, Any]:
        """Read one explicitly selected authored joint without inferring its role.

        Args:
            entity_id: Exact semantic articulation identity in the registration.
            joint: Exact native joint name, preserving asset-defined spelling.

        Returns:
            Owned position/acquisition data; it carries no endpoint verdict.
        """
        self._check_episode()
        self._port.validate_observation_scope(self._bridge.clock)
        uid, native = self._native(entity_id, articulated=True)
        names = tuple(native.joint_names)
        if type(joint) is not str or names.count(joint) != 1:
            raise ValueError(
                "Joint observations require one exact declared native joint."
            )
        import torch

        values = native.get_qpos()
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 2
            or values.shape[1] != len(names)
        ):
            raise ValueError(
                "Articulation position columns differ from its joint names."
            )
        result = self._packet(
            entity_id,
            uid,
            values[:, names.index(joint)],
            kind="joint_position",
            observation_kind="final",
            joint=joint,
        )
        self._port.validate_observation_scope(self._bridge.clock)
        return result

    def capture_recorded_rigid_pose(
        self, entity_id: str, frame_indices: Sequence[int]
    ) -> dict[str, Any]:
        """Copy exact historical poses from the canonical raw observation owner.

        Args:
            entity_id: Declared semantic rigid-object identity.
            frame_indices: Exact seeded state slot for every native row.

        Returns:
            Owned pose/acquisition data preserving raw availability and temporal
            identity. Invalid samples remain unavailable; no live pose is read.
        """
        from embodichain.utils.math import xyz_quat_to_4x4_matrix

        self._check_episode()
        uid, owner = self._rigid_owners[entity_id]
        if owner is None:
            raise ValueError("The rigid entity had no bound native owner.")
        if self._observations is None:
            raise RuntimeError("Observation history is unavailable.")
        candidates = [
            name
            for name, descriptor in self._observations.schema.items()
            if descriptor.entity_uid == uid
            and descriptor.quantity == "pose"
            and descriptor.coordinate_frame == "arena"
            and descriptor.representation == "xyz_xyzw"
            and descriptor.shape == (7,)
            and descriptor.units == ("m", "unitless")
        ]
        if not candidates:
            raise RuntimeError(f"Recorded raw arena pose for {uid!r} is unavailable.")
        field = candidates[0]
        snapshot = self._observations.read_history(
            frame_indices, fields=(field,), episode_ids=self._episodes
        )
        result = self._packet(
            entity_id,
            uid,
            xyz_quat_to_4x4_matrix(snapshot.values[field]),
            kind="rigid_pose",
            observation_kind="recorded",
            frame_indices=snapshot.frame_indices,
            source_valid=snapshot.valid[field],
            source_errors=snapshot.acquisition_errors[field],
        )
        result["recorded_acquisition"] = {
            "field_id": self._observations.field_ids[field],
            "env_ids": list(snapshot.env_ids),
            "episode_ids": list(snapshot.episode_ids),
            "frame_indices": list(snapshot.frame_indices),
            "observation_revisions": list(snapshot.revisions),
            "timestamps": list(snapshot.timestamps),
            "clock_domain": snapshot.clock_domain,
        }
        return result

    def capture_recorded_joint_policy(
        self,
        entity_id: str,
        *,
        preset_id: str,
        segment_id: str,
        source_path: Sequence[str | int],
        frame_indices: Sequence[int],
    ) -> dict[str, Any]:
        """Copy an executed joint-policy receipt without reading current qpos.

        Args:
            entity_id: Exact registered semantic articulation identity.
            preset_id: Explicit named joint policy used by this occurrence.
            segment_id: Exact compiled segment name.
            source_path: Full compiled segment occurrence path.
            frame_indices: Its exact completed boundary slot for every row.

        Returns:
            Original policy result, raw acquisition, boundary and episode scope.
            Missing policy execution remains unavailable; current qpos is never
            substituted for a historical measurement.
        """
        from .stability import SimulationJointStabilityConditionCfg

        self._check_episode()
        cfg = self._registration.settle_presets[preset_id]
        if (
            type(cfg) is not SimulationJointStabilityConditionCfg
            or cfg.entity != entity_id
        ):
            raise ValueError(
                "The selected preset is not the declared articulation joint policy."
            )
        uid, owner = self._joint_owners[entity_id]
        if owner is None:
            raise ValueError(
                "The articulation had no native owner when collection was bound."
            )
        segments = [
            segment
            for segment in self._bridge.program.iter_segments()
            if segment.name == segment_id and segment.source_path == tuple(source_path)
        ]
        if len(segments) != 1:
            raise ValueError(
                "Recorded joint evidence requires one exact compiled occurrence."
            )
        segment = segments[0]
        policies = [
            policy for policy in segment.post_policies if policy.cfg.preset == preset_id
        ]
        if len(policies) != 1:
            raise ValueError(
                "Recorded joint evidence requires one exact post-policy occurrence."
            )
        receipt = deepcopy(
            dict(self._port.post_policy_metadata(policies[0], segment=segment))
        )
        channel = receipt.get("joint_observation")
        if (
            not isinstance(channel, Mapping)
            or len(frame_indices) != len(self._episodes)
            or any(
                type(frame) is not int or frame != channel.get("end_step")
                for frame in frame_indices
            )
        ):
            raise ValueError(
                "Recorded joint evidence has no raw acquisition at this exact boundary."
            )
        self._check_episode()
        return dict(
            schema_version="simulation.state_observation/v1",
            entity_id=entity_id,
            native_uid=uid,
            joint=cfg.joint,
            observation_kind="recorded",
            episode_ids=list(self._episodes),
            boundary_frames=list(frame_indices),
            policy_result=receipt,
        )
