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
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, TypeAlias

import torch

from embodichain.lab.evaluation.observations import _measure_near_target_frame
from embodichain.lab.gym.envs.managers.observation_manager import _ObservationCapture
from embodichain.lab.gym.envs.managers.observations import _read_entity_state

if TYPE_CHECKING:
    from embodichain.lab.evaluation import NearTargetHoldSpec
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.task_program.integrations.catalog import (
        SimulationTaskProgramRegistration,
    )

__all__: list[str] = []

_FinalRules: TypeAlias = tuple[tuple[str, "NearTargetHoldSpec"], ...]


@dataclass(frozen=True, slots=True)
class _FinalMeasurement:
    satisfied: torch.Tensor
    conditions: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]
    revision: int
    timestamp: float | None


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
    frame = capture.finish(None)
    conditions = tuple(_measure_near_target_frame(spec, frame) for _, spec in rules)
    return _FinalMeasurement(
        torch.stack([passed for passed, _, _ in conditions]).all(dim=0),
        conditions,
        frame.revision,
        frame.timestamp,
    )


def _final_acceptance_rows(
    rules: _FinalRules,
    episode_ids: tuple[int, ...],
    accepted: tuple[bool, ...],
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
            "program_accepted": program[index],
            "physical_satisfied": physical[index],
            "observation_revision": (
                None if measurement is None else measurement.revision
            ),
            "timestamp": None if measurement is None else measurement.timestamp,
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
