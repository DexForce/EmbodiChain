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

"""Owned values for fixed-scene trajectory generation, independent of hosts.

Tensor inputs are detached and copied at construction. Frozen dataclasses prevent
field replacement; consumers must still treat their owned tensors as read-only.
Positions use the complete declared joint order. All times are in seconds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Literal

import torch

__all__ = [
    "SceneCase",
    "MotionSnapshot",
    "TrajectoryPhase",
    "TrajectoryTemplate",
    "CandidateIdentity",
    "CandidateTrajectoryBatch",
    "ValidationCheck",
    "ValidationResult",
    "ExpertEpisode",
    "CommitReceipt",
]


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _names(values: tuple[str, ...], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of names")
    result = tuple(values)
    for value in result:
        _text(value, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _tensor(value: torch.Tensor, name: str, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain finite floating point values")
    return value.detach().clone()


def _pose(value: torch.Tensor, name: str) -> torch.Tensor:
    result = _tensor(value, name, 2)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    # Float32 transforms accumulated by a host can carry small roundoff errors.
    check = result.to(dtype=torch.float64)
    rotation = check[:3, :3]
    identity = torch.eye(3, dtype=check.dtype, device=check.device)
    bottom = check.new_tensor([0, 0, 0, 1])
    if (
        not torch.allclose(check[3], bottom, atol=1e-5, rtol=0)
        or not torch.allclose(rotation.T @ rotation, identity, atol=1e-5, rtol=0)
        or not torch.allclose(
            torch.linalg.det(rotation), check.new_tensor(1), atol=1e-5, rtol=0
        )
    ):
        raise ValueError(f"{name} must be a proper SE(3) transform")
    return result


def _poses(values: Mapping[str, torch.Tensor], name: str) -> Mapping[str, torch.Tensor]:
    result = {}
    for key, value in values.items():
        _text(key, name)
        result[key] = _pose(value, f"{name}[{key}]")
    return MappingProxyType(result)


def _json(value: object) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_json(item) for item in value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            _text(key, "metadata key")
            result[key] = _json(item)
        return MappingProxyType(result)
    raise ValueError("metadata must contain only finite JSON values")


def _phases(
    values: tuple[TrajectoryPhase, ...], length: int
) -> tuple[TrajectoryPhase, ...]:
    result = tuple(values)
    previous_stop = 0
    seen = set()
    for phase in result:
        if not isinstance(phase, TrajectoryPhase):
            raise ValueError("phases must contain TrajectoryPhase values")
        if (
            phase.phase_id in seen
            or phase.start_index < previous_stop
            or phase.stop_index > length
        ):
            raise ValueError(
                "phases must be unique, ordered, nonoverlapping, and within valid samples"
            )
        seen.add(phase.phase_id)
        previous_stop = phase.stop_index
    return result


@dataclass(frozen=True)
class SceneCase:
    """A fixed scene and allowed initial state, with no live environment references."""

    scene_case_id: str
    initial_state_id: str
    scene_signature: str
    task_id: str
    robot_profile_id: str
    calibration_id: str = "default"

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class MotionSnapshot:
    """One copied complete robot state; poses map local frames into the world."""

    scene_case: SceneCase
    joint_names: tuple[str, ...]
    joint_positions: torch.Tensor
    joint_velocities: torch.Tensor
    root_pose: torch.Tensor
    entity_poses: Mapping[str, torch.Tensor] = field(default_factory=dict)
    dependency_revisions: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.scene_case, SceneCase):
            raise ValueError("scene_case must be a SceneCase")
        names = _names(self.joint_names, "joint_names")
        positions = _tensor(self.joint_positions, "joint_positions", 1)
        velocities = _tensor(self.joint_velocities, "joint_velocities", 1)
        if (
            not names
            or positions.shape != (len(names),)
            or velocities.shape != positions.shape
        ):
            raise ValueError(
                "robot positions and velocities must match the complete joint_names"
            )
        if velocities.device != positions.device:
            raise ValueError("robot positions and velocities must share a device")
        revisions = dict(self.dependency_revisions)
        for key, revision in revisions.items():
            _text(key, "dependency revision key")
            _integer(revision, "dependency revision")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "joint_positions", positions)
        object.__setattr__(self, "joint_velocities", velocities)
        object.__setattr__(self, "root_pose", _pose(self.root_pose, "root_pose"))
        object.__setattr__(
            self, "entity_poses", _poses(self.entity_poses, "entity_poses")
        )
        object.__setattr__(self, "dependency_revisions", MappingProxyType(revisions))


@dataclass(frozen=True)
class TrajectoryPhase:
    """An explicitly annotated half-open sample interval ``[start, stop)``.

    Phase endpoints are samples ``start_index`` and ``stop_index - 1``.
    Unannotated samples remain unchanged by augmentation operators.
    """

    phase_id: str
    start_index: int
    stop_index: int
    kind: Literal["free", "contact", "hold"] = "free"
    allowed_operators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.phase_id, "phase_id")
        _integer(self.start_index, "start_index")
        _integer(self.stop_index, "stop_index", self.start_index + 1)
        if self.kind not in ("free", "contact", "hold"):
            raise ValueError("phase kind must be free, contact, or hold")
        object.__setattr__(
            self,
            "allowed_operators",
            _names(self.allowed_operators, "allowed_operators"),
        )


@dataclass(frozen=True)
class TrajectoryTemplate:
    """A full-joint qpos reference with arrival intervals and explicit permissions.

    ``dt[0]`` is zero; every later interval is strictly positive. Empty phases
    and operator permissions permit replay only. EEF conversion is owned by
    an injected planning adapter before constructing this qpos contract.
    """

    source_id: str
    source_revision: str
    template_id: str
    joint_names: tuple[str, ...]
    positions: torch.Tensor
    dt: torch.Tensor
    phases: tuple[TrajectoryPhase, ...] = ()
    allowed_operators: tuple[str, ...] = ()
    validator_id: str = "default"
    controlled_joint_indices: tuple[int, ...] = ()
    representation: str = "qpos"

    def __post_init__(self) -> None:
        for name in ("source_id", "source_revision", "template_id", "validator_id"):
            _text(getattr(self, name), name)
        if self.representation != "qpos":
            raise ValueError("only explicit qpos templates are supported")
        names = _names(self.joint_names, "joint_names")
        positions = _tensor(self.positions, "positions", 2)
        dt = _tensor(self.dt, "dt", 1)
        if not names or positions.shape[0] < 1 or positions.shape[1] != len(names):
            raise ValueError("positions must have shape (N >= 1, len(joint_names))")
        if dt.shape != positions.shape[:1] or dt.device != positions.device:
            raise ValueError("dt must match the positions length and device")
        if dt[0] != 0 or bool((dt[1:] <= 0).any()):
            raise ValueError("dt[0] must be zero and later arrival intervals positive")
        controlled = tuple(self.controlled_joint_indices)
        for index in controlled:
            _integer(index, "controlled joint index")
            if index >= len(names):
                raise ValueError("controlled joint index is outside joint_names")
        if len(set(controlled)) != len(controlled):
            raise ValueError("controlled joint indices must be unique")
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "phases", _phases(self.phases, positions.shape[0]))
        object.__setattr__(
            self,
            "allowed_operators",
            _names(self.allowed_operators, "allowed_operators"),
        )
        object.__setattr__(self, "controlled_joint_indices", controlled)


@dataclass(frozen=True)
class CandidateIdentity:
    """Candidate lineage independent of execution slots and their epochs."""

    scene_case_id: str
    initial_state_id: str
    candidate_id: str
    geometry_family_id: str
    source_id: str
    source_revision: str
    template_id: str
    attempt_id: int = 0
    parent_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "scene_case_id",
            "initial_state_id",
            "candidate_id",
            "geometry_family_id",
            "source_id",
            "source_revision",
            "template_id",
        ):
            _text(getattr(self, name), name)
        _integer(self.attempt_id, "attempt_id")
        if self.parent_id is not None:
            _text(self.parent_id, "parent_id")
            if self.parent_id == self.candidate_id:
                raise ValueError("a candidate cannot be its own parent")


@dataclass(frozen=True)
class CandidateTrajectoryBatch:
    """Logical candidate rows with safe padding and no physical batch assumption.

    ``positions`` is ``(C,N,D_full)``, ``dt`` is ``(C,N)``, and each
    ``valid_length`` is in ``[1,N]``. An empty candidate set uses ``C=0``.
    Padded positions hold the final valid sample and padded intervals are zero.
    They must never be executed or recorded; use :attr:`valid_mask`.
    """

    positions: torch.Tensor
    dt: torch.Tensor
    valid_length: torch.Tensor
    identities: tuple[CandidateIdentity, ...]
    joint_names: tuple[str, ...]
    phases: tuple[tuple[TrajectoryPhase, ...], ...] = ()
    factors: tuple[Mapping[str, float | int | str], ...] = ()
    source_row_indices: torch.Tensor | None = None

    def __post_init__(self) -> None:
        positions = _tensor(self.positions, "positions", 3)
        dt = _tensor(self.dt, "dt", 2)
        count, horizon, joints = positions.shape
        names = _names(self.joint_names, "joint_names")
        if not names or joints != len(names) or (count and horizon < 1):
            raise ValueError(
                "positions must use complete joint_names and a nonempty horizon"
            )
        if dt.shape != (count, horizon) or dt.device != positions.device:
            raise ValueError("dt must match positions shape and device")
        lengths = self.valid_length
        if (
            not isinstance(lengths, torch.Tensor)
            or lengths.shape != (count,)
            or lengths.dtype != torch.int64
        ):
            raise ValueError("valid_length must be an int64 tensor of shape (C,)")
        lengths = lengths.detach().clone().to(device=positions.device)
        if bool(((lengths < 1) | (lengths > horizon)).any()):
            raise ValueError("valid_length must be between 1 and the padded horizon")
        identities = tuple(self.identities)
        if len(identities) != count or not all(
            isinstance(value, CandidateIdentity) for value in identities
        ):
            raise ValueError("identities must contain one CandidateIdentity per row")
        keys = [(value.candidate_id, value.attempt_id) for value in identities]
        if len(set(keys)) != len(keys):
            raise ValueError("candidate attempts must be unique within a batch")
        phases = tuple(self.phases) if self.phases else ((),) * count
        if len(phases) != count:
            raise ValueError("phases must contain one entry per candidate")
        normalized_phases = []
        for row, length in enumerate(lengths.tolist()):
            if dt[row, 0] != 0 or bool((dt[row, 1:length] <= 0).any()):
                raise ValueError(
                    "valid dt must start at zero and use positive arrival intervals"
                )
            if bool((dt[row, length:] != 0).any()):
                raise ValueError("padded dt must be zero")
            if not torch.equal(
                positions[row, length:],
                positions[row, length - 1].expand(horizon - length, joints),
            ):
                raise ValueError("padded positions must hold the final valid sample")
            normalized_phases.append(_phases(phases[row], length))
        factors = tuple(self.factors) if self.factors else ({},) * count
        if len(factors) != count:
            raise ValueError("factors must contain one entry per candidate")
        copied_factors = []
        for row_factors in factors:
            copied = dict(row_factors)
            for key, value in copied.items():
                _text(key, "factor name")
                if type(value) not in (float, int, str) or (
                    isinstance(value, (float, int)) and not math.isfinite(value)
                ):
                    raise ValueError("factor values must be finite numbers or strings")
            copied_factors.append(MappingProxyType(copied))
        indices = self.source_row_indices
        if indices is not None:
            if (
                not isinstance(indices, torch.Tensor)
                or indices.shape != (count,)
                or indices.dtype != torch.int64
                or bool((indices < 0).any())
            ):
                raise ValueError(
                    "source_row_indices must be nonnegative int64 values of shape (C,)"
                )
            indices = indices.detach().clone()
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "valid_length", lengths)
        object.__setattr__(self, "joint_names", names)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(self, "phases", tuple(normalized_phases))
        object.__setattr__(self, "factors", tuple(copied_factors))
        object.__setattr__(self, "source_row_indices", indices)

    @property
    def valid_mask(self) -> torch.Tensor:
        """Return a fresh boolean mask selecting actual samples in each row.

        Returns:
            A boolean tensor of shape ``(C, N)`` that excludes padded samples.
        """
        return (
            torch.arange(self.positions.shape[1], device=self.positions.device)[None, :]
            < self.valid_length[:, None]
        )

    def row(self, index: int) -> CandidateTrajectoryBatch:
        """Copy one candidate and its aligned metadata, retaining padded shape.

        Args:
            index: Nonnegative candidate row index.

        Returns:
            An owned single-row batch with shape ``(1, N, D_full)`` and aligned metadata.
        """
        _integer(index, "row index")
        if index >= len(self.identities):
            raise IndexError("candidate row is out of range")
        return CandidateTrajectoryBatch(
            positions=self.positions[index : index + 1],
            dt=self.dt[index : index + 1],
            valid_length=self.valid_length[index : index + 1],
            identities=(self.identities[index],),
            joint_names=self.joint_names,
            phases=(self.phases[index],),
            factors=(self.factors[index],),
            source_row_indices=(
                None
                if self.source_row_indices is None
                else self.source_row_indices[index : index + 1]
            ),
        )


@dataclass(frozen=True)
class ValidationCheck:
    """One mandatory check; unavailable and not-run checks cannot accept data."""

    check_id: str
    status: Literal["not_run", "passed", "failed", "unavailable"]
    detail: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.check_id, "check_id")
        if self.status not in ("not_run", "passed", "failed", "unavailable"):
            raise ValueError("unknown validation status")
        if not isinstance(self.detail, str):
            raise ValueError("validation detail must be text")
        metrics = dict(self.metrics)
        for key, value in metrics.items():
            _text(key, "metric key")
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("validation metrics must be finite numbers")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))


@dataclass(frozen=True)
class ValidationResult:
    """A fail-closed collection of required validation checks."""

    checks: tuple[ValidationCheck, ...]

    def __post_init__(self) -> None:
        checks = tuple(self.checks)
        if not all(isinstance(check, ValidationCheck) for check in checks):
            raise ValueError("checks must contain ValidationCheck values")
        if len({check.check_id for check in checks}) != len(checks):
            raise ValueError("validation check IDs must be unique")
        object.__setattr__(self, "checks", checks)

    @property
    def accepted(self) -> bool:
        """Return whether every required check was run and passed.

        Returns:
            ``True`` only for a nonempty result whose checks all have passing status.
        """
        return bool(self.checks) and all(
            check.status == "passed" for check in self.checks
        )


@dataclass(frozen=True)
class ExpertEpisode:
    """Frozen rollout evidence with causal ``obs[t], action[t], obs[t+1]`` pairs.

    Observations and timestamps contain ``T+1`` samples, including the terminal
    observation. Actions contain the ``T`` commands actually submitted. This
    contract checks structure; hosts remain responsible for measuring evidence.
    """

    identity: CandidateIdentity
    observations: Mapping[str, torch.Tensor]
    actions: torch.Tensor
    timestamps: torch.Tensor
    action_representation: str
    validation: ValidationResult
    episode_id: str
    commit_id: str
    phases: tuple[TrajectoryPhase, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, CandidateIdentity) or not isinstance(
            self.validation, ValidationResult
        ):
            raise ValueError(
                "episode identity and validation must use their value contracts"
            )
        _text(self.episode_id, "episode_id")
        _text(self.commit_id, "commit_id")
        _text(self.action_representation, "action_representation")
        actions = _tensor(self.actions, "actions", 2)
        times = _tensor(self.timestamps, "timestamps", 1)
        steps = actions.shape[0]
        if steps < 1 or actions.shape[1] < 1 or times.shape != (steps + 1,):
            raise ValueError("episodes require T >= 1 actions and T+1 timestamps")
        if times[0] < 0 or bool((times[1:] <= times[:-1]).any()):
            raise ValueError("timestamps must be nonnegative and strictly increasing")
        if not self.observations:
            raise ValueError("episodes must contain actual observations")
        observations = {}
        for key, value in self.observations.items():
            _text(key, "observation key")
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim < 1
                or value.shape[0] != steps + 1
            ):
                raise ValueError(
                    "each observation must contain T+1 samples including terminal"
                )
            if value.is_complex() or not bool(torch.isfinite(value).all()):
                raise ValueError("observations must contain finite real values")
            observations[key] = value.detach().clone()
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "timestamps", times)
        object.__setattr__(self, "observations", MappingProxyType(observations))
        object.__setattr__(self, "phases", _phases(self.phases, steps + 1))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", _json(self.metadata))


@dataclass(frozen=True)
class CommitReceipt:
    """Sink confirmation tied to stable episode/candidate IDs, never a slot.

    The idempotent ``commit_id`` remains fixed across persistence retries.
    ``submission_id`` identifies an individual write attempt so a delayed
    failure cannot invalidate a newer retry of the same episode.
    """

    episode_id: str
    candidate_id: str
    attempt_id: int
    storage_id: str
    commit_id: str
    scene_case_id: str
    confirmed: bool = True
    error: str = ""
    submission_id: int = 0

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "candidate_id",
            "storage_id",
            "commit_id",
            "scene_case_id",
        ):
            _text(getattr(self, name), name)
        _integer(self.attempt_id, "attempt_id")
        _integer(self.submission_id, "submission_id")
        if type(self.confirmed) is not bool:
            raise ValueError("confirmed must be a boolean")
        if not isinstance(self.error, str) or (self.confirmed and self.error):
            raise ValueError("only unconfirmed receipts may contain an error")
