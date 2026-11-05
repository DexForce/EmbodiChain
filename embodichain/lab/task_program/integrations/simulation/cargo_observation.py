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

"""Episode-scoped Cargo observations through physical declarations and recording APIs."""

from __future__ import annotations
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, NoReturn, TYPE_CHECKING
import torch
from embodichain.utils import configclass
from embodichain.lab.sim.evidence import PhysicalEvidenceRequest
from embodichain.lab.sim.scene.source_geometry import read_source_vertices
from embodichain.lab.sim.sensors.cargo_envelope import (
    CargoEnvelopeObservation,
    capture_carrier_contents,
    observe_carrier_contents,
)

from ._recorded_observations import _recorded_pose_window

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.envs.managers.observation_manager import ObservationManager
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge

__all__ = ["SimulationCargoEnvelopeCfg", "SimulationCargoEnvelopeCollector"]


@configclass
class SimulationCargoEnvelopeCfg:
    """Explicit native selections; physical mesh/pose values stay with the environment.

    Attributes:
        carrier_uids: Native rigid carriers selected by the host declaration.
        excluded_uids: Intentionally manipulated objects excluded from initial cargo.
    """

    carrier_uids: tuple[str, ...] = ()
    excluded_uids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("carrier_uids", "excluded_uids"):
            values = getattr(self, name)
            if (
                type(values) is not tuple
                or len(set(values)) != len(values)
                or any(
                    type(value) is not str or not value or value != value.strip()
                    for value in values
                )
            ):
                raise ValueError(f"{name} requires unique exact native identities.")

    def snapshot(self) -> SimulationCargoEnvelopeCfg:
        """Own and validate the complete immutable selection values."""
        result = deepcopy(self)
        result.validate()
        result.__post_init__()
        return result


class _DeclaredMeshes(Mapping[str, torch.Tensor]):
    def __init__(self, sources: Mapping[str, Mapping[str, Any]]) -> None:
        self._sources = deepcopy(dict(sources))

    def __iter__(self) -> Iterator[str]:
        return iter(self._sources)

    def __len__(self) -> int:
        return len(self._sources)

    def __getitem__(self, key: str) -> torch.Tensor:
        # Source capture reads each carrier then only nonexcluded candidate meshes.
        return torch.as_tensor(read_source_vertices(self._sources[key]))


class _RecordedPoses(dict[str, dict[str, Any]]):
    def __init__(self) -> None:
        super().__init__()
        self.errors: dict[str, str] = {}

    def __getitem__(self, key: str) -> dict[str, Any]:
        if key in self.errors:
            raise ValueError(self.errors[key])
        return super().__getitem__(key)


class _UnavailableCounts(Sequence[int]):
    def __init__(self, count: int, reason: str) -> None:
        self.count, self.reason = count, reason

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, row: int | slice) -> NoReturn:
        raise ValueError(self.reason)


class SimulationCargoEnvelopeCollector:
    """Borrow one actual bridge/environment and own its initial Cargo facts.

    No task graph, planner, sensor or TaskSpec evaluator is constructed. Recorded
    poses come only from the canonical episode-qualified raw owner; native
    poses remain current terminal measurements. Missing recording stays unavailable.
    """

    def __init__(
        self,
        environment: EmbodiedEnv,
        bridge: TaskProgramDemoBridge,
        cfg: SimulationCargoEnvelopeCfg,
        *,
        observations: ObservationManager | None = None,
    ) -> None:
        """Capture source contents before the first command of this actual bridge.

        Args:
            environment: Initialized owner of the physical declarations/recording.
            bridge: Actual currently selected bridge at control step zero.
            cfg: Explicit carrier/exclusion native identities.
            observations: Actual canonical owner of the recorded raw pose fields.
        """
        if type(cfg) is not SimulationCargoEnvelopeCfg:
            raise TypeError("Cargo collection requires its typed declaration.")
        if (
            environment.active_task_program_bridge is not bridge
            or bridge.clock.step_index != 0
        ):
            raise ValueError(
                "Cargo collection must bind the active bridge before its first step."
            )
        episodes = environment.episode_context.snapshot()
        if len(episodes) != environment.num_envs or not all(
            row["initialized"] for row in episodes
        ):
            raise ValueError("Cargo collection requires initialized physical episodes.")
        if observations is not None:
            observations.validate_history_owner(environment.episode_context)
        self._observations = observations
        self._episode_owner = environment.episode_context
        self._native_owners: dict[str, object] = {}
        self._recorded_acquisition: dict[str, Any] = {}
        self._env, self._bridge, self._clock = environment, bridge, bridge.clock
        self._sim = environment.sim
        self._episodes = tuple(row["episode"] for row in episodes)
        self._cfg = cfg.snapshot()
        self._closed = False
        sources = {item.uid: item.to_dict() for item in environment.cfg.rigid_object}
        if not set(self._cfg.carrier_uids) <= sources.keys():
            raise ValueError("Cargo carriers must name physical rigid declarations.")
        initial_request = self._request()
        self._bindings = capture_carrier_contents(
            _DeclaredMeshes(sources),
            self._pose,
            initial_request,
            episode_ids=self._episodes,
            carrier_ids=self._cfg.carrier_uids,
            excluded_ids=self._cfg.excluded_uids,
        )
        self._guard_request(initial_request)

    def _guard(self) -> None:
        if self._closed or getattr(self._env, "_closed", False):
            raise RuntimeError("Cargo collection is closed.")
        if (
            self._env.active_task_program_bridge is not self._bridge
            or self._bridge.clock is not self._clock
            or self._env.sim is not self._sim
        ):
            raise ValueError("Cargo collection belongs to another runtime owner.")
        if self._env.episode_context is not self._episode_owner:
            raise ValueError("Cargo collection belongs to another episode owner.")
        current = self._episode_owner.snapshot()
        if tuple(row["episode"] for row in current) != self._episodes or not all(
            row["initialized"] for row in current
        ):
            raise ValueError("Cargo collection belongs to another episode.")

    def _guard_request(self, request: PhysicalEvidenceRequest) -> None:
        self._guard()
        for uid, owner in self._native_owners.items():
            if (
                self._sim.get_rigid_object(uid) is not owner
                or getattr(owner, "uid", None) != uid
            ):
                raise ValueError(
                    "Cargo rigid observation belongs to another native owner."
                )
        if (
            request.timestamp != self._clock.now()
            or request.observation_revision != self._clock.step_index
        ):
            raise ValueError(
                "Cargo execution clock advanced during observation capture."
            )

    def _request(self) -> PhysicalEvidenceRequest:
        self._guard()
        return PhysicalEvidenceRequest(
            self._clock.now(),
            self._clock.step_index,
            torch.arange(len(self._episodes), dtype=torch.long),
        )

    def _pose(self, uid: str) -> torch.Tensor:
        self._guard()
        entity = self._sim.get_rigid_object(uid)
        if entity is None:
            raise ValueError(f"Cargo rigid observation is unavailable: {uid}")
        if (
            getattr(entity, "uid", None) != uid
            or self._native_owners.setdefault(uid, entity) is not entity
        ):
            raise ValueError("Cargo rigid observation belongs to another native owner.")
        return entity.get_local_pose(to_matrix=True)

    def capture(self, *, invocation_id: str) -> CargoEnvelopeObservation:
        """Read exact complete recorded prefixes and current terminal poses.

        Args:
            invocation_id: Actual execution occurrence at this terminal boundary.

        Returns:
            Existing physical evidence channels with native episode/row and
            original first-failure indices. Recording read errors become invalid
            acquisition rows; they never turn into an empty successful history.
        """
        request = self._request()
        trajectory = _RecordedPoses()
        acquisitions = {}
        try:
            if self._observations is None:
                raise RuntimeError("Observation history is unavailable.")
            self._observations.validate_history_owner(self._episode_owner)
            counts = self._observations.read_history_step_counts(
                episode_ids=self._episodes
            )
            if len(counts) != len(self._episodes) or any(
                type(n) is not int or n < 0 for n in counts
            ):
                raise ValueError("Cargo recording has invalid completed prefixes.")
            identifiers = tuple(
                dict.fromkeys(
                    uid
                    for value in self._bindings
                    for uid in (value.carrier_id, value.object_id)
                )
            )
            for uid in identifiers:
                try:
                    rows, metadata = _recorded_pose_window(
                        self._observations,
                        uid,
                        (0,) * len(counts),
                        tuple(max(0, n - 1) for n in counts),
                        episode_ids=self._episodes,
                    )
                    trajectory[uid] = {"pose": rows}
                    acquisitions[uid] = metadata
                except (
                    RuntimeError,
                    ValueError,
                    KeyError,
                    AttributeError,
                    TypeError,
                ) as error:
                    trajectory.errors[uid] = str(error) or type(error).__name__
        except (RuntimeError, ValueError, AttributeError, TypeError) as error:
            counts = _UnavailableCounts(
                len(self._episodes), str(error) or type(error).__name__
            )
        result = observe_carrier_contents(
            self._bindings,
            self._pose,
            request,
            episode_ids=self._episodes,
            invocation_id=invocation_id,
            trajectory=trajectory,
            step_counts=counts,
            trajectory_pose_format="matrix",
        )
        self._guard_request(request)
        self._recorded_acquisition = acquisitions
        return result

    def capture_report(self, *, invocation_id: str) -> dict[str, Any]:
        """Serialize owned episode facts without selecting a task success verdict.

        Args:
            invocation_id: Actual terminal execution occurrence.

        Returns:
            Native participants, original prefix indices and existing physical
            channel validity/errors. Unavailable observed values serialize as None.
        """
        observation = self.capture(invocation_id=invocation_id)
        contents = []
        for binding in self._bindings:
            key = binding.evidence_id
            channel = observation.frame.get(key)
            values = channel.values.detach().cpu().tolist()
            valid = channel.valid.detach().cpu().tolist()
            contents.append(
                dict(
                    carrier=binding.carrier_id,
                    object=binding.object_id,
                    initial_mask=observation.initial_masks[key].detach().cpu().tolist(),
                    observed_mask=[
                        value if available else None
                        for value, available in zip(values, valid, strict=True)
                    ],
                    valid_mask=valid,
                    acquisition_errors=list(channel.acquisition_errors),
                    first_failed_frames=list(observation.first_failed_frames[key]),
                    evidence_id=key,
                )
            )
        return dict(
            schema_version="lab.cargo_envelope_observation/v1",
            scope=observation.scope,
            episode_ids=list(observation.episode_ids),
            invocation_id=observation.invocation_id,
            env_ids=observation.frame.env_ids.detach().cpu().tolist(),
            initial_timestamp=observation.initial_timestamp,
            timestamp=observation.frame.timestamp,
            observation_revision=observation.frame.observation_revision,
            clock_domain="execution_step_clock",
            recorded_acquisition=deepcopy(self._recorded_acquisition),
            contents=contents,
        )

    def close(self) -> None:
        """Invalidate this observer without closing any borrowed runtime resource."""
        self._closed = True
