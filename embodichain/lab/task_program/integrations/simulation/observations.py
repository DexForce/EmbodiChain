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
    from embodichain.lab.gym.utils.trajectory_state import RecordedTrajectoryReader
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
    from ..catalog import SimulationTaskProgramRegistration

__all__ = ["SimulationSupportObservationCollector"]


class SimulationSupportObservationCollector:
    """Read physical support snapshots from one explicit environment/bridge pair.

    Construct before the first bridge control step, using the actual bridge rather
    than an adapter template port. This object borrows the existing policy
    session, registry and sensors. It creates no native resources, steps no
    physics and computes no TaskSpec predicate or release decision.

    Args:
        environment: Initialized recording environment for this execution.
        bridge: Exact bridge whose clock and policy session drive this run.
        registration: Sealed declaration used to create the runtime.
        recording: Optional read-only trajectory reader supplied by the recording
            host. Historical capture requires this explicit dependency.
    """

    def __init__(
        self,
        environment: EmbodiedEnv,
        bridge: TaskProgramDemoBridge,
        registration: SimulationTaskProgramRegistration,
        *,
        recording: RecordedTrajectoryReader | None = None,
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
        if recording is not None:
            recording.validate_episode_source(environment.episode_context)
        self._env, self._bridge, self._port = environment, bridge, port
        self._registration = registration
        self._recording = recording
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
        return {
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
        if self._recording is None:
            raise RuntimeError("Recorded trajectory reader is unavailable.")
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
        poses = {}
        for name in (cfg.entity, cfg.reference):
            poses[name] = self._recording.read_rigid_poses(
                self._bindings[name].simulation_uid,
                frame_indices,
                episode_ids=self._episodes,
            ).poses.tolist()
        self._check_episode()
        acquisition["clock_domain"] = (
            "environment_step"
            if cfg.support_contract_revision == 1
            else "simulation_time"
        )
        return {
            "native_support": native,
            "support_acquisition": acquisition,
            "boundary_poses": poses,
            "boundary_frames": list(frame_indices),
            "episode_ids": list(self._episodes),
        }
