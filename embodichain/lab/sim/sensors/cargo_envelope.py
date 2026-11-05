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

"""Owned carrier-content envelope observations, independent of normative verdicts."""

from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal
import torch
from embodichain.compute.geometry.cargo import (
    inside_carrier_envelope,
    first_carrier_envelope_failure,
)
from embodichain.lab.sim.evidence import (
    PhysicalEvidenceRequest,
    PhysicalEvidenceBatch,
    PhysicalEvidenceFrame,
)

__all__ = [
    "CargoEnvelopeBinding",
    "CargoEnvelopeObservation",
    "capture_carrier_contents",
    "observe_carrier_contents",
]

PoseReader = Callable[[str], torch.Tensor]


@dataclass(frozen=True, slots=True, eq=False)
class CargoEnvelopeBinding:
    """Owned initial content facts for an explicitly selected carrier/object pair."""

    carrier_id: str
    object_id: str
    vertices: torch.Tensor
    bounds: torch.Tensor
    initial_mask: torch.Tensor
    env_ids: torch.Tensor
    timestamp: float
    episode_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("carrier_id", "object_id"):
            value = getattr(self, name)
            if type(value) is not str or not value or value != value.strip():
                raise ValueError("Cargo identities must be exact nonempty strings.")
        if self.carrier_id == self.object_id:
            raise ValueError("Cargo and carrier must be distinct.")
        if (
            self.vertices.ndim != 2
            or self.vertices.shape[1] != 3
            or not len(self.vertices)
            or not self.vertices.is_floating_point()
            or not torch.isfinite(self.vertices).all()
            or self.bounds.shape != (2, 3)
            or not torch.isfinite(self.bounds).all()
            or (self.bounds[0] > self.bounds[1]).any()
            or self.initial_mask.dtype != torch.bool
        ):
            raise ValueError("Cargo bindings require finite geometry and boolean rows.")
        request = PhysicalEvidenceRequest(self.timestamp, 0, self.env_ids)
        if (
            type(self.episode_ids) is not tuple
            or len(self.episode_ids) != len(request.env_ids)
            or any(type(value) is not int or value < 0 for value in self.episode_ids)
        ):
            raise ValueError("Cargo bindings require actual episode identities.")
        if self.initial_mask.shape != request.env_ids.shape:
            raise ValueError("Cargo initial mask differs from its environment rows.")
        for name in ("vertices", "bounds", "initial_mask", "env_ids"):
            object.__setattr__(self, name, getattr(self, name).detach().clone())

    @property
    def evidence_id(self) -> str:
        """Return an unambiguous channel ID derived from both exact identities."""
        import json

        return "cargo.envelope:" + json.dumps(
            [self.carrier_id, self.object_id], ensure_ascii=False, separators=(",", ":")
        )


@dataclass(frozen=True, slots=True, eq=False)
class CargoEnvelopeObservation:
    """Necessary envelope facts with row, episode, occurrence and window identity.

    A valid True value only describes the mesh envelope. It is neither a cavity
    containment nor a physical retention certificate. Invalid rows retain their
    acquisition error, and must not be interpreted as measured failures/successes.
    """

    frame: PhysicalEvidenceFrame
    initial_masks: Mapping[str, torch.Tensor]
    first_failed_frames: Mapping[str, tuple[int | None, ...]]
    episode_ids: tuple[int, ...]
    invocation_id: str
    initial_timestamp: float
    scope: str

    def __post_init__(self) -> None:
        if (
            type(self.invocation_id) is not str
            or not self.invocation_id
            or self.invocation_id != self.invocation_id.strip()
            or type(self.episode_ids) is not tuple
            or len(self.episode_ids) != self.frame.batch_size
            or any(type(value) is not int or value < 0 for value in self.episode_ids)
            or self.scope not in {"terminal", "recorded_trajectory_and_terminal"}
            or self.frame.timestamp < self.initial_timestamp
        ):
            raise ValueError("Cargo evidence requires exact execution/window identity.")
        keys = set(self.frame.batches)
        if set(self.initial_masks) != keys or set(self.first_failed_frames) != keys:
            raise ValueError("Cargo observation metadata differs from its channels.")
        object.__setattr__(self, "frame", self.frame.snapshot())
        object.__setattr__(
            self,
            "initial_masks",
            MappingProxyType(
                {
                    key: value.detach().clone()
                    for key, value in self.initial_masks.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "first_failed_frames",
            MappingProxyType(dict(self.first_failed_frames)),
        )


def _pose(
    read_pose: PoseReader, name: str, request: PhysicalEvidenceRequest
) -> torch.Tensor:
    value = read_pose(name)
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or value.shape != (len(request.env_ids), 4, 4)
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"Cargo observed pose unavailable: {name}")
    return value


def capture_carrier_contents(
    vertices: Mapping[str, torch.Tensor],
    read_pose: PoseReader,
    request: PhysicalEvidenceRequest,
    *,
    episode_ids: tuple[int, ...],
    carrier_ids: Sequence[str],
    excluded_ids: Sequence[str] = (),
) -> tuple[CargoEnvelopeBinding, ...]:
    """Capture initially enclosed objects from explicit owner geometry and poses.

    Args:
        vertices: Ordered source object-local mesh arrays (V, 3), in source units.
        read_pose: Current physical pose reader, keyed by the same logical IDs.
        request: Initial observation clock and stable environment row identities.
        episode_ids: Actual native episode identity per environment row.
        carrier_ids: Explicit carrier selections, visited in source sorted order.
        excluded_ids: Explicitly manipulated objects, excluded from cargo capture.

    Returns:
        Owned bindings in source carrier/object order. Missing initial facts raise
        instead of silently dropping possible contents. No simulator is created.
    """

    def mesh_points(value: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape[1] != 3
            or not len(value)
            or not torch.isfinite(value).all()
        ):
            raise ValueError("Cargo initial geometry is unavailable.")
        return value

    carriers = set(carrier_ids)
    excluded = set(excluded_ids)
    result = []
    for carrier_id in sorted(carriers):
        carrier_pose = _pose(read_pose, carrier_id, request)
        mesh = mesh_points(vertices[carrier_id]).to(carrier_pose)
        bounds = torch.stack((mesh.amin(0), mesh.amax(0)))
        for object_id in vertices:
            if object_id in carriers or object_id in excluded:
                continue
            points = mesh_points(vertices[object_id]).to(carrier_pose)
            initial = inside_carrier_envelope(
                points,
                _pose(read_pose, object_id, request).to(points),
                carrier_pose,
                bounds,
            )
            if initial.any():
                result.append(
                    CargoEnvelopeBinding(
                        carrier_id,
                        object_id,
                        points,
                        bounds,
                        initial,
                        request.env_ids,
                        request.timestamp,
                        episode_ids,
                    )
                )
    return tuple(result)


def observe_carrier_contents(
    bindings: tuple[CargoEnvelopeBinding, ...],
    read_pose: PoseReader,
    request: PhysicalEvidenceRequest,
    *,
    episode_ids: tuple[int, ...],
    invocation_id: str,
    trajectory: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
    step_counts: Sequence[int] | torch.Tensor | None = None,
    trajectory_pose_format: Literal["xyz_xyzw", "matrix"] = "xyz_xyzw",
) -> CargoEnvelopeObservation:
    """Observe terminal and optional full-prefix envelope facts without TaskSpec.

    Args:
        bindings: Owned initial content facts from this execution.
        read_pose: Fresh terminal physical pose reader.
        request: Terminal time, revision and unchanged environment identities.
        episode_ids: Actual native episode identity per environment row.
        invocation_id: Actual execution occurrence owning this observation.
        trajectory: Optional recorded rigid-object pose arrays (B, T, 7).
        step_counts: Exact valid recorded prefix length per environment.
        trajectory_pose_format: Explicit raw samples or public recorded matrices.

    Returns:
        Existing physical evidence frame plus initial/window metadata. Missing
        trajectories/poses remain unavailable, with acquisition errors. The
        original 64-frame grid and first violation are preserved.
    """
    batches, masks, first_frames = {}, {}, {}
    if type(bindings) is not tuple or any(
        type(value) is not CargoEnvelopeBinding for value in bindings
    ):
        raise TypeError("Cargo observation requires immutable typed bindings.")
    initial_time = request.timestamp
    for binding in bindings:
        if (
            not torch.equal(binding.env_ids.to(request.env_ids), request.env_ids)
            or request.timestamp < binding.timestamp
            or episode_ids != binding.episode_ids
        ):
            raise ValueError(
                "Cargo observation changed initial rows/episodes or regressed time."
            )
        initial_time = min(initial_time, binding.timestamp)
        key = binding.evidence_id
        if key in batches:
            raise ValueError("Duplicate cargo envelope binding.")
        initial = binding.initial_mask.to(request.env_ids.device)
        values = torch.ones_like(initial)
        valid = torch.ones_like(initial)
        errors = [None] * len(initial)
        first = [None] * len(initial)
        try:
            object_pose = read_pose(binding.object_id).to(binding.vertices)
            carrier_pose = read_pose(binding.carrier_id).to(binding.vertices)
            if (
                object_pose.shape != (len(initial), 4, 4)
                or carrier_pose.shape != object_pose.shape
            ):
                raise ValueError("Cargo terminal pose batch is unavailable.")
            for row in initial.nonzero().flatten().tolist():
                try:
                    if (
                        not torch.isfinite(object_pose[row]).all()
                        or not torch.isfinite(carrier_pose[row]).all()
                    ):
                        raise ValueError(
                            "Cargo terminal pose contains nonfinite values."
                        )
                    values[row] = inside_carrier_envelope(
                        binding.vertices,
                        object_pose[row : row + 1],
                        carrier_pose[row : row + 1],
                        binding.bounds,
                    )[0]
                except (ValueError, RuntimeError) as error:
                    valid[row] = False
                    errors[row] = str(error) or type(error).__name__
        except (ValueError, RuntimeError, KeyError, AttributeError, TypeError) as error:
            valid[initial] = False
            for row in initial.nonzero().flatten().tolist():
                errors[row] = str(error) or type(error).__name__
        if trajectory is not None:
            for row in initial.nonzero().flatten().tolist():
                try:
                    if step_counts is None or len(step_counts) != len(initial):
                        raise ValueError(
                            "Cargo trajectory requires per-environment step counts."
                        )
                    count = int(step_counts[row])
                    first[row] = first_carrier_envelope_failure(
                        binding.vertices,
                        trajectory[binding.object_id]["pose"][row],
                        trajectory[binding.carrier_id]["pose"][row],
                        binding.bounds,
                        frame_count=count,
                        pose_format=trajectory_pose_format,
                    )
                    if first[row] is not None:
                        values[row] = False
                except (
                    ValueError,
                    RuntimeError,
                    KeyError,
                    IndexError,
                    AttributeError,
                    TypeError,
                ) as error:
                    valid[row] = False
                    errors[row] = str(error) or type(error).__name__
        batches[key] = PhysicalEvidenceBatch(
            key,
            values,
            valid,
            tuple(errors),
            request.timestamp,
            request.env_ids,
            request.observation_revision,
        )
        masks[key] = initial
        first_frames[key] = tuple(first)
    return CargoEnvelopeObservation(
        PhysicalEvidenceFrame(
            request.timestamp, request.observation_revision, request.env_ids, batches
        ),
        masks,
        first_frames,
        episode_ids,
        invocation_id,
        initial_time,
        "terminal" if trajectory is None else "recorded_trajectory_and_terminal",
    )
