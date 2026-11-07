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

"""Fresh physical measurements for deployment-owned final acceptance."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, TYPE_CHECKING, TypeAlias

import torch

from embodichain.data_pipeline.recording import stable_config_hash
from embodichain.lab.evaluation.observations import (
    _measure_near_target_frame,
    _rigid_pose_field,
)
from embodichain.lab.gym.envs.managers.observation_manager import _ObservationCapture
from embodichain.lab.gym.envs.managers.observation_types import (
    ObservationDescriptor,
    ObservationFrame,
    ObservationSample,
)
from embodichain.lab.gym.envs.managers.observations import _read_entity_state

if TYPE_CHECKING:
    from embodichain.lab.evaluation import NearTargetHoldSpec
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.task_program.integrations.catalog import (
        SimulationTaskProgramRegistration,
    )

__all__: list[str] = []

_FinalRules: TypeAlias = tuple[tuple[str, "NearTargetHoldSpec"], ...]
_FINAL_OBSERVATIONS_KEY = "final_acceptance_observations"


@dataclass(frozen=True, slots=True)
class _FinalMeasurement:
    satisfied: torch.Tensor
    conditions: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    frame: ObservationFrame


def _deployment_final_rules(
    registration: SimulationTaskProgramRegistration | None,
) -> _FinalRules:
    return tuple(
        (ref, registration.evaluation_conditions[ref])
        for ref in getattr(registration, "evaluation_final_refs", ())
    )


def _measure_final_conditions(
    env: EmbodiedEnv, rules: _FinalRules
) -> _FinalMeasurement:
    fields = dict.fromkeys(field for _, spec in rules for field in spec.required_fields)
    owner = env.episode_context
    timestamp = getattr(env.sim, "simulation_time", None)
    with _ObservationCapture(
        owner.episode_ids,
        float(timestamp) if isinstance(timestamp, (int, float)) else None,
        episode_owner=owner,
    ) as capture:
        for field in fields:
            cfg = env._evaluation_recording[f"recording:{field}"]
            capture.fields[field] = _read_entity_state(
                env, cfg.params["entity_cfg"].uid, "pose"
            )
    return _evaluate_final_frame(rules, capture.finish(None))


def _evaluate_final_frame(
    rules: _FinalRules, frame: ObservationFrame
) -> _FinalMeasurement:
    conditions = tuple(_measure_near_target_frame(spec, frame) for _, spec in rules)
    return _FinalMeasurement(
        torch.stack([passed for passed, _, _ in conditions]).all(dim=0),
        conditions,
        frame,
    )


def _final_acceptance_rows(
    rules: _FinalRules,
    episode_ids: tuple[int, ...],
    accepted: tuple[bool, ...],
    execution_eligible: tuple[bool, ...],
    completion: torch.Tensor | None,
    measurement: _FinalMeasurement | None,
) -> list[dict[str, Any]]:
    program = (
        completion.tolist() if completion is not None else [False] * len(episode_ids)
    )
    physical = (
        measurement.satisfied.tolist()
        if measurement is not None
        else [None] * len(episode_ids)
    )
    rows = [
        {
            "episode_id": episode_id,
            "accepted": accepted[index],
            "execution_eligible": execution_eligible[index],
            "program_accepted": program[index],
            "physical_satisfied": physical[index],
            "observation_revision": (
                None if measurement is None else measurement.frame.revision
            ),
            "timestamp": None if measurement is None else measurement.frame.timestamp,
            "conditions": {},
        }
        for index, episode_id in enumerate(episode_ids)
    ]
    if measurement is None:
        for row in rows:
            row["conditions"] = {
                ref: {
                    "status": "not_run",
                    "distance": None,
                    "tolerance": spec.tolerance,
                }
                for ref, spec in rules
            }
        return rows
    for (ref, spec), (passed, valid, distance) in zip(
        rules, measurement.conditions, strict=True
    ):
        for row, passed_row, available, measured in zip(
            rows, passed.tolist(), valid.tolist(), distance.tolist(), strict=True
        ):
            row["conditions"][ref] = {
                "status": (
                    ("satisfied" if passed_row else "failed")
                    if available
                    else "unavailable"
                ),
                "distance": measured if available and math.isfinite(measured) else None,
                "tolerance": spec.tolerance,
            }
    return rows


def _export_final_observations(
    rules: _FinalRules,
    frames: Mapping[int, ObservationFrame],
    episodes: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    if not frames:
        return None
    rows = []
    for episode in episodes:
        env_id = episode["env_id"]
        frame = frames.get(env_id)
        if frame is None:
            rows.append(None)
            continue
        fields = {}
        for field, sample in frame.fields.items():
            descriptor = asdict(sample.descriptor)
            descriptor["dtype"] = str(sample.descriptor.dtype)
            fields[field] = {
                "descriptor": descriptor,
                "values": sample.values[env_id : env_id + 1].to("cpu", copy=True),
                "valid": sample.valid[env_id : env_id + 1].to("cpu", copy=True),
                "acquisition_errors": (sample.acquisition_errors[env_id],),
            }
        rows.append(
            {
                "env_id": env_id,
                "episode_uuid": episode["episode_uuid"],
                "episode_id": frame.episode_ids[env_id],
                "observation_revision": frame.revision,
                "timestamp": frame.timestamp,
                "fields": fields,
            }
        )
    if not any(row is not None for row in rows):
        return None
    return {"schema_version": 1, "rules_hash": stable_config_hash(rules), "rows": rows}


def _replay_final_acceptance(
    rules: _FinalRules, trajectory: Mapping[str, Any]
) -> tuple[dict[str, Any], ...] | None:
    if not rules:
        return None
    meta = trajectory["meta"]
    count = int(meta["num_envs"])
    episodes = meta.get("demo_episodes")
    if not isinstance(episodes, (list, tuple)):
        episodes = ()
    if _FINAL_OBSERVATIONS_KEY not in trajectory:
        return tuple(
            _replay_final_row(
                rules, None, episodes[index] if index < len(episodes) else {}
            )
            for index in range(count)
        )

    payload = trajectory[_FINAL_OBSERVATIONS_KEY]
    if (
        not isinstance(payload, Mapping)
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or not isinstance(payload.get("rows"), list)
        or len(payload["rows"]) != count
        or len(episodes) != count
        or any(not isinstance(episode, Mapping) for episode in episodes)
        or meta.get("env_ids") != [episode.get("env_id") for episode in episodes]
    ):
        raise ValueError("Invalid final acceptance observations in trajectory.")
    if payload.get("rules_hash") != stable_config_hash(rules):
        raise ValueError(
            "Trajectory final acceptance rules differ from the deployment."
        )
    return tuple(
        _replay_final_row(rules, row, episode)
        for row, episode in zip(payload["rows"], episodes, strict=True)
    )


def _replay_final_row(
    rules: _FinalRules, row: Mapping[str, Any] | None, episode: Mapping[str, Any]
) -> dict[str, Any]:
    seal = episode.get("final_acceptance") if isinstance(episode, Mapping) else None
    if not isinstance(seal, Mapping):
        seal = {}
    recorded = seal.get("accepted")
    if row is None:
        not_run = recorded is False and (
            "observation_revision" in seal and seal["observation_revision"] is None
        )
        return {
            "status": "not_run" if not_run else "not_recorded",
            "accepted": False if not_run else None,
            "recorded_accepted": recorded,
        }

    if (
        not isinstance(row, Mapping)
        or type(recorded) is not bool
        or type(seal.get("observation_revision")) is not int
        or type(seal.get("episode_id")) is not int
        or type(seal.get("execution_eligible")) is not bool
        or type(seal.get("program_accepted")) is not bool
        or row.get("env_id") != episode.get("env_id")
        or row.get("episode_uuid") != episode.get("episode_uuid")
        or row.get("episode_id") != seal["episode_id"]
        or row.get("observation_revision") != seal["observation_revision"]
        or row.get("timestamp") != seal.get("timestamp")
    ):
        raise ValueError(
            "Final acceptance observation does not match the episode seal."
        )

    expected_fields = {field for _, spec in rules for field in spec.required_fields}
    raw_fields = row.get("fields")
    if not isinstance(raw_fields, Mapping) or set(raw_fields) != expected_fields:
        raise ValueError("Final acceptance observation fields differ from the rules.")
    fields = {}
    for field, sample in raw_fields.items():
        values = sample["values"]
        descriptor = dict(sample["descriptor"])
        if (
            not isinstance(values, torch.Tensor)
            or values.shape != (1, 7)
            or descriptor["dtype"] != str(values.dtype)
            or _rigid_pose_field(descriptor["entity_uid"]) != field
        ):
            raise ValueError(f"Invalid final acceptance observation for {field!r}.")
        descriptor["dtype"] = values.dtype
        fields[field] = ObservationSample(
            ObservationDescriptor(**descriptor),
            values,
            sample["valid"],
            sample["acquisition_errors"],
        )
    measurement = _evaluate_final_frame(
        rules,
        ObservationFrame(
            None,
            fields,
            (seal["episode_id"],),
            seal["observation_revision"],
            seal.get("timestamp"),
        ),
    )
    eligible = seal["execution_eligible"]
    completion = torch.tensor([seal["program_accepted"]], dtype=torch.bool)
    accepted = eligible and seal["program_accepted"] and bool(measurement.satisfied[0])
    result = _final_acceptance_rows(
        rules,
        measurement.frame.episode_ids,
        (accepted,),
        (eligible,),
        completion,
        measurement,
    )[0]
    return {"status": "recomputed", "recorded_accepted": recorded, **result}
