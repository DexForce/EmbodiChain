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

"""Project shared evidence onto Task Program subgoal boundaries."""

from __future__ import annotations

import torch

from embodichain.lab.evaluation import EvaluationStatus as Status
from embodichain.lab.evaluation.observations import _rule_report
from ._final_acceptance import _evaluate_frame_rule

__all__: list[str] = []


def _statuses(active, passed, *, otherwise=Status.FAILED):
    status = torch.full_like(active, Status.NOT_RUN, dtype=torch.int8)
    status[active] = otherwise
    status[active & passed] = Status.SATISFIED
    return status


def _held_status(evidence, state, active):
    held = torch.zeros_like(active)
    if state is None:
        return _statuses(active, held, otherwise=Status.UNAVAILABLE)
    for relation in (
        *state.held_objects.values(),
        *state.coordinated_held_objects.values(),
    ):
        if relation.semantics.entity_id == evidence.ref:
            held |= relation.env_mask
    return _statuses(active, held if evidence.kind == "held" else ~held)


def _evidence_report(evidence, status, details=None):
    codes = [Status(value) for value in status.tolist()]
    return {
        "kind": evidence.kind,
        "ref": evidence.ref,
        "required": evidence.required,
        "assurance": (
            "symbolic" if evidence.kind in ("held", "released") else evidence.assurance
        ),
        "status": [code.name.lower() for code in codes],
        "reasons": [
            (
                "unsupported_capability"
                if evidence.kind == "unsupported" and code is Status.UNAVAILABLE
                else code.reason
            )
            for code in codes
        ],
        "evidence": details,
    }


def _unrun_outcomes(contract, rows):
    status = torch.full_like(rows, Status.NOT_RUN, dtype=torch.int8)
    return [_evidence_report(entry, status) for entry in contract.outcomes]


def _check_prerequisites(evidence, active, env_ids, task_state, capture, episode_owner):
    fields = {
        field
        for entry in evidence
        if entry.spec is not None
        for field in entry.spec.required_fields
    }
    frame = capture(tuple(sorted(fields))) if fields and capture is not None else None
    if frame is not None and (
        episode_owner is None
        or frame.episode_owner is not episode_owner
        or frame.episode_ids != episode_owner.episode_ids
        or frame.revision < 0
        or any(row < 0 or row >= len(frame.episode_ids) for row in env_ids.tolist())
    ):
        frame = None
    measurements = {}
    reports = []
    accepted = active.clone()
    for entry in evidence:
        detail = None
        if entry.kind in ("held", "released"):
            status = _held_status(entry, task_state, active)
        elif frame is None:
            status = _statuses(
                active, torch.zeros_like(active), otherwise=Status.UNAVAILABLE
            )
        else:
            result = _evaluate_frame_rule(
                entry.ref, entry.spec, frame, active.device, measurements, rows=env_ids
            )
            detail = {
                "evaluation_identity": entry.spec.identity,
                "observation_revision": frame.revision,
                "episode_ids": [frame.episode_ids[row] for row in env_ids.tolist()],
                "result": _rule_report(result),
            }
            status = result.status
            status[~active] = Status.NOT_RUN
        reports.append(_evidence_report(entry, status, detail))
        accepted &= status == Status.SATISFIED
    return accepted, reports


def _call_effect_status(entry, result, active, call_index):
    call = next((call for call in result.calls if call.call_index == call_index), None)
    if call is None:
        return (
            _statuses(active, torch.zeros_like(active), otherwise=Status.NOT_RUN),
            None,
        )
    status = _statuses(active & call.entered_mask, call.completed_mask)
    detail = {"call_index": call.call_index, "semantic_id": call.semantic_id}
    if entry.assurance == "verified":
        verified = torch.zeros_like(active)
        effects = []
        for effect in call.effects:
            if effect.boundary_kind == "terminal":
                verified |= effect.success_mask
                effects.append(
                    {
                        "verification_id": effect.verification_id,
                        "observation_revision": effect.observation_revision,
                        "monitor_id": effect.monitor_id,
                    }
                )
        status[active & call.completed_mask & ~verified] = Status.UNAVAILABLE
        detail["effects"] = effects
    return status, detail


def _check_outcomes(contract, result, active, validators, policies):
    accepted = active.clone()
    reports = []
    for entry in contract.outcomes:
        detail = None
        if entry.kind in ("held", "released"):
            status = _held_status(
                entry, result.task_state, active & result.success_mask
            )
        elif entry.kind == "unsupported":
            status = _statuses(
                active, torch.zeros_like(active), otherwise=Status.UNAVAILABLE
            )
        elif entry.kind == "call_effect":
            status, detail = _call_effect_status(entry, result, active, entry.ref)
        else:
            traces = validators if entry.kind == "validator" else policies
            if entry.ref >= len(traces):
                status = _statuses(
                    active, torch.zeros_like(active), otherwise=Status.NOT_RUN
                )
            else:
                detail = traces[entry.ref]
                passed = torch.tensor(
                    detail["result_mask"], device=active.device, dtype=torch.bool
                )
                status = _statuses(active & result.success_mask, passed)
                rule = detail.get("result")
                if entry.kind == "validator" and detail["kind"] == "condition" and rule:
                    status = torch.tensor(
                        [Status[name.upper()] for name in rule["status"]],
                        device=active.device,
                        dtype=torch.int8,
                    )
                    status[~active] = Status.NOT_RUN
        reports.append(_evidence_report(entry, status, detail))
        if entry.required:
            accepted &= status == Status.SATISFIED
    return accepted, reports
