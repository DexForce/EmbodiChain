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

"""Physical grip effect interpretation from shared sensor evidence."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from copy import deepcopy
import math
from typing import Any
import torch
from embodichain.compute.geometry.grip_contacts import classify_contacts
from embodichain.lab.sim.evidence import PhysicalEvidenceRequest, PhysicalEvidenceBatch
from embodichain.lab.sim.sensors.contact_sampling import (
    ContactEvidenceSampler,
    CONTACT_ACTOR_IDS,
    CONTACT_VALID_MASK,
    CONTACT_NORMALS,
    CONTACT_IMPULSES,
)

__all__ = ["GripContactObserver"]
_INVALID_CONTACT = (
    "Target grip contact is incomplete, one-sided, non-opposing, or saturated."
)
_RELEASE_SCOPE = (
    "experimental_open_hand_without_bilateral_grip_not_contact_free_separation"
)


def _identifier(value: str, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty exact identifier.")


def _finite_trace_copy(value: Any, path: str, errors: list[dict[str, str]]) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        label = (
            "NaN" if math.isnan(value) else "+Infinity" if value > 0 else "-Infinity"
        )
        errors.append({"path": path, "value": label})
        return {"nonfinite": label}
    if isinstance(value, torch.Tensor):
        return _finite_trace_copy(value.detach().cpu().tolist(), path, errors)
    if isinstance(value, dict):
        return {
            key: _finite_trace_copy(item, f"{path}.{key}", errors)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, deque)):
        return [
            _finite_trace_copy(item, f"{path}[{index}]", errors)
            for index, item in enumerate(value)
        ]
    return deepcopy(value)


class GripContactObserver:
    """Observe target-specific bilateral contact through physical evidence frames.

    Revision 1 requires opposed positive-impulse pad contacts or a measured clear
    release. Revision 2 additionally permits the explicitly configured open-hand
    observation for one-sided release; it never certifies acquisition, bilateral
    release, or geometric separation.
    """

    def __init__(
        self,
        sampler: ContactEvidenceSampler,
        object_actors: Mapping[str, torch.Tensor],
        pad_actors: Mapping[str, torch.Tensor],
        *,
        sensor_uid: str,
        algorithm_revision: int = 1,
        trace_capacity: int = 4096,
        release_opening: Callable[[str, int], Mapping[str, object]] | None = None,
    ) -> None:
        """Own raw actor bindings and bounded observer-lifetime audit state.

        Args:
            sampler: Instance-owned contact acquisition source.
            object_actors: One native actor per object and environment, shape (B, 1).
            pad_actors: Two explicitly declared pad actors per control part, (B, 2).
            sensor_uid: Exact configured sensor identity, retained in audit metadata.
            algorithm_revision: Physical interpretation revision, either 1 or 2.
            trace_capacity: Maximum retained query records.
            release_opening: Required only for explicit revision-2 release observation.
        """
        _identifier(sensor_uid, "sensor_uid")
        if type(algorithm_revision) is not int or algorithm_revision not in (1, 2):
            raise ValueError("Unsupported native grip algorithm revision.")
        if type(trace_capacity) is not int or trace_capacity < 1:
            raise ValueError("trace_capacity must be a positive integer.")
        if algorithm_revision == 2 and not callable(release_opening):
            raise ValueError("Grip revision 2 requires an explicit opening reader.")
        self.sampler, self.sensor_uid = sampler, sensor_uid
        self.algorithm_revision = algorithm_revision
        self._release_opening = release_opening if algorithm_revision == 2 else None
        self._object_actors = self._actors(object_actors, sampler.num_envs, 1)
        self._pad_actors = self._actors(pad_actors, sampler.num_envs, 2)
        self.trace: deque[dict[str, Any]] = deque(maxlen=trace_capacity)
        self.cumulative_query_count = 0
        self.recorded_query_count = 0
        self.evicted_query_count = 0
        self.nonfinite_query_count = 0
        self.dropped_query_count = 0
        self.maximum_dropped_contacts = 0

    @staticmethod
    def _actors(
        bindings: Mapping[str, torch.Tensor], count: int, width: int
    ) -> dict[str, torch.Tensor]:
        if not bindings:
            raise ValueError("Native grip actor bindings must not be empty.")
        result = {}
        for name, value in bindings.items():
            _identifier(name, "actor binding")
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype not in (torch.int32, torch.int64)
                or value.shape != (count, width)
                or bool((value < 0).any())
            ):
                raise ValueError(
                    "Native grip actor IDs have invalid shape, dtype or identity."
                )
            if width == 2 and bool((value[:, 0] == value[:, 1]).any()):
                raise ValueError("Native grip pads must be two distinct actors.")
            result[name] = value.detach().cpu().clone()
        return result

    def snapshot_trace(self) -> dict[str, Any]:
        """Return raw trace data without sampling or claiming task success.

        The frontend owns any application JSON schema wrapper. Field meanings and
        observer-lifetime loss/nonfinite accounting retain the original contract.
        """
        nonfinite: list[dict[str, str]] = []
        records = _finite_trace_copy(self.trace, "trace", nonfinite)
        return {
            "source": "native_contact_sensor",
            "observed": self.recorded_query_count > 0,
            "sensor_uid": self.sensor_uid,
            "scope": "observer_lifetime_raw_queries_not_task_success",
            "actor_bindings": {
                "scope": "observer_lifetime_native_ids_not_contact_evidence",
                "env_axis": "native_env_id",
                "object_actor_ids": {
                    key: values.tolist() for key, values in self._object_actors.items()
                },
                "pad_actor_ids": {
                    key: values.tolist() for key, values in self._pad_actors.items()
                },
            },
            "cumulative_query_count": self.cumulative_query_count,
            "recorded_query_count": self.recorded_query_count,
            "retained_query_count": len(self.trace),
            "evicted_query_count": self.evicted_query_count,
            "unrecorded_query_count": self.cumulative_query_count
            - self.recorded_query_count,
            "trace_capacity": self.trace.maxlen,
            "overflowed": self.evicted_query_count > 0,
            "trace_complete": self.cumulative_query_count
            == self.recorded_query_count
            == len(self.trace)
            and self.evicted_query_count == 0,
            "dropped_contacts": {
                "queries_with_drops": self.dropped_query_count,
                "maximum_per_query": self.maximum_dropped_contacts,
            },
            "finite_audit": {
                "valid": self.nonfinite_query_count == 0 and not nonfinite,
                "nonfinite_query_count": self.nonfinite_query_count,
                "retained_nonfinite_locations": nonfinite,
                "nonfinite_encoding": "explicit_value_marker_at_original_location",
            },
            "trace": records,
        }

    def observe(
        self,
        request: PhysicalEvidenceRequest,
        *,
        object_id: str,
        control_part: str,
        expected_contact: bool,
        releasing: bool = False,
    ) -> PhysicalEvidenceBatch:
        """Measure one bound physical grip without a Task Program query.

        Args:
            request: Observation time, revision and ordered native environment IDs.
            object_id: Explicit object identity installed on this observer.
            control_part: Explicit calibrated hand binding.
            expected_contact: Requested physical contact value for the audit trace.
            releasing: Enable the selected revision's release interpretation.

        Returns:
            Bilateral-contact values with independent validity and errors.
            Neither an invalid sample nor this result certifies task success.
        """
        if type(request) is not PhysicalEvidenceRequest:
            raise TypeError("request must be PhysicalEvidenceRequest.")
        if request.channels and request.channels != ("grip.bilateral",):
            raise ValueError(
                "Grip observations support only the grip.bilateral channel."
            )
        if type(expected_contact) is not bool or type(releasing) is not bool:
            raise TypeError(
                "Contact expectation and release selection must be booleans."
            )
        context = request
        uid, part = object_id, control_part
        if uid not in self._object_actors or part not in self._pad_actors:
            raise ValueError(
                "Native grip query is outside the declared actor bindings."
            )
        self.cumulative_query_count += 1
        sample = self.sampler.sample(
            PhysicalEvidenceRequest(
                context.timestamp,
                context.observation_revision,
                context.env_ids.detach().cpu(),
                (
                    CONTACT_VALID_MASK,
                    CONTACT_ACTOR_IDS,
                    CONTACT_NORMALS,
                    CONTACT_IMPULSES,
                ),
            )
        )
        batches = sample.frame.batches
        if (
            sample.acquisition_error is not None
            and batches[CONTACT_VALID_MASK].values.shape[1] == 0
        ):
            return PhysicalEvidenceBatch(
                "grip.bilateral",
                torch.zeros(
                    len(context.env_ids),
                    dtype=torch.bool,
                    device=context.env_ids.device,
                ),
                torch.zeros(
                    len(context.env_ids),
                    dtype=torch.bool,
                    device=context.env_ids.device,
                ),
                (sample.acquisition_error,) * len(context.env_ids),
                context.timestamp,
                context.env_ids,
                context.observation_revision,
            )
        values, valid, errors, records = [], [], [], []
        available_rows = (
            torch.stack(tuple(batch.valid for batch in batches.values()))
            .all(0)
            .tolist()
        )
        finite_query = True
        for row, env_id in enumerate(sample.frame.env_ids.tolist()):
            mask = batches[CONTACT_VALID_MASK].values[row]
            trace_pairs = batches[CONTACT_ACTOR_IDS].values[row, mask]
            trace_normals = batches[CONTACT_NORMALS].values[row, mask]
            trace_impulses = batches[CONTACT_IMPULSES].values[row, mask]
            # Host records already need CPU ownership. Keep public Frame checks
            # there, but preserve the original backend arithmetic at physical
            # thresholds by evaluating only the packed contacts on their source.
            compute_device = sample.source_device or trace_normals.device
            pairs = trace_pairs.to(compute_device)
            normals = trace_normals.to(compute_device)
            impulses = trace_impulses.to(compute_device)
            finite = bool(
                torch.isfinite(normals).all() and torch.isfinite(impulses).all()
            )
            finite_query &= finite
            target = int(self._object_actors[uid][env_id, 0])
            pads = tuple(self._pad_actors[part][env_id].tolist())
            value, ok = classify_contacts(
                pairs, normals, impulses, target=target, pads=pads
            )
            opening = None
            available = available_rows[row]
            if (
                self._release_opening is not None
                and expected_contact is False
                and releasing
                and not value
                and not ok
                and finite
                and sample.dropped_contacts == 0
                and available
                and bool((torch.linalg.vector_norm(normals, dim=-1) > 1e-8).all())
                and bool((impulses >= 0).all())
            ):
                touching = [
                    bool(
                        (
                            (
                                ((pairs[:, 0] == target) & (pairs[:, 1] == pad))
                                | ((pairs[:, 1] == target) & (pairs[:, 0] == pad))
                            )
                            & (impulses > 0)
                        ).any()
                    )
                    for pad in pads
                ]
                if sum(touching) == 1:
                    opening = dict(self._release_opening(part, env_id))
                    finite &= opening["finite"] is True
                    finite_query &= finite
                    ok = opening["open"] is True and finite
            ok &= sample.dropped_contacts == 0 and available
            values.append(value)
            valid.append(ok)
            errors.append(None if ok else _INVALID_CONTACT)
            selected = (trace_pairs == target).any(-1)
            records.append(
                {
                    "env_id": env_id,
                    "value": value,
                    "valid": ok,
                    "finite": finite,
                    "error": errors[-1],
                    "pairs": trace_pairs[selected].tolist(),
                    "normals": trace_normals[selected].tolist(),
                    "impulses": trace_impulses[selected].tolist(),
                    "raw_contacts": {
                        "pairs": trace_pairs.tolist(),
                        "normals": trace_normals.tolist(),
                        "impulses": trace_impulses.tolist(),
                    },
                    **({"release_opening": opening} if opening is not None else {}),
                }
            )
        if len(self.trace) == self.trace.maxlen:
            self.evicted_query_count += 1
        self.recorded_query_count += 1
        self.nonfinite_query_count += int(not finite_query)
        self.dropped_query_count += int(sample.dropped_contacts > 0)
        self.maximum_dropped_contacts = max(
            self.maximum_dropped_contacts, sample.dropped_contacts
        )
        self.trace.append(
            {
                "query_index": self.cumulative_query_count,
                "timestamp": context.timestamp,
                "observation_revision": context.observation_revision,
                "object_id": uid,
                "control_part": part,
                "expected": expected_contact,
                "dropped_contacts": sample.dropped_contacts,
                "rows": records,
            }
        )
        return PhysicalEvidenceBatch(
            "grip.bilateral",
            torch.tensor(values, dtype=torch.bool, device=context.env_ids.device),
            torch.tensor(valid, dtype=torch.bool, device=context.env_ids.device),
            tuple(errors),
            context.timestamp,
            context.env_ids,
            context.observation_revision,
        )
