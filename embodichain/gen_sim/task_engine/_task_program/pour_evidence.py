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
"""Measured E3 pose-proxy evidence, never a fluid-transfer observation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

import torch

from .grip_evidence import classify_contacts
from .held_evidence import delta, matrix, one

__all__: list[str] = []


@dataclass(frozen=True)
class PourProxyContract:
    """One resolved invocation target and explicitly declared proxy tolerances."""

    source_uid: str
    receiver_uid: str
    resource_id: str
    invocation_id: str
    invocation_revision: int
    env_id: int
    planned_at: float
    start_step: int
    end_step: int
    step_dt: float
    entry_pose: torch.Tensor
    peak_pose: torch.Tensor
    pivot_object: tuple[float, float, float]
    receiver_anchor_local: tuple[float, float, float]
    tilt_tolerance_rad: float
    restore_tolerance_rad: float
    position_tolerance_m: float


@dataclass(frozen=True)
class NativePourWindow:
    """One row of contiguous, same-recording native body poses, including end."""

    source_uid: str
    receiver_uid: str
    env_id: int
    steps: tuple[int, ...]
    source_poses: torch.Tensor
    receiver_poses: torch.Tensor


@dataclass(frozen=True)
class PourHoldEvidence:
    """Existing call/effect traces and native grip queries; no resampling."""

    plan_attempts: tuple[Mapping[str, Any], ...]
    effects: tuple[Mapping[str, Any], ...]
    native_grips: tuple[Mapping[str, Any], ...]
    object_actor_id: int
    pad_actor_ids: tuple[int, int]
    grasp_control_part: str


def _poses(value: Any, count: int) -> torch.Tensor:
    if torch.as_tensor(value).is_complex():
        raise ValueError("Complex pose evidence is unavailable.")
    return matrix(value, count)


def _clock(value: Any) -> float:
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid physical observation clock.")
    return float(value)


def _hold(c: PourProxyContract, h: PourHoldEvidence) -> tuple[bool, dict[str, Any]]:
    attempts = sorted(h.plan_attempts, key=lambda a: _clock(a["planned_at"]))
    times = [_clock(a["planned_at"]) for a in attempts]
    if len(set(times)) != len(times):
        raise ValueError("Ambiguous plan-attempt timestamps.")
    attempt = one(a for a in attempts if a["planned_at"] == c.planned_at)
    if (attempt["invocation_id"], attempt["invocation_revision"]) != (
        c.invocation_id,
        c.invocation_revision,
    ) or not any(v is True for v in attempt["plan_success_mask"]):
        raise ValueError("Plan attempt belongs to another invocation.")
    following = next((t for t in times if t > c.planned_at), math.inf)
    start, end = c.start_step * c.step_dt, c.end_step * c.step_dt
    if not c.planned_at <= start < end < following:
        raise ValueError("Pour window crosses a plan-attempt boundary.")
    actors = (h.object_actor_id, *h.pad_actor_ids)
    if (
        len(actors) != 3
        or len(set(actors)) != 3
        or any(type(a) is not int for a in actors)
    ):
        raise ValueError("Exact distinct native object/pad actors are required.")
    selected, lost = [], False
    for effect in h.effects:
        spec = effect["effect_spec"]
        if (
            spec["invocation_id"] != c.invocation_id
            or spec["invocation_revision"] != c.invocation_revision
        ):
            continue
        time = _clock(effect["timestamp"])
        boundary = effect["boundary"]
        if (
            not c.planned_at <= time < following
            or boundary.get("segment_name", "pour") != "pour"
        ):
            continue
        kind = boundary["kind"]
        if kind not in {"phase_effect_gate", "in_flight_guard", "terminal"}:
            continue
        if kind != "terminal" and boundary.get("segment_name") != "pour":
            raise ValueError("Physical guard has no exact Pour phase identity.")
        monitor = effect["monitor"]
        if (spec["semantic_id"], spec["skill_id"], spec["effect_kind"]) != (
            "simulation.pour",
            "pour",
            "attach",
        ) or (monitor["monitor_id"], monitor["revision"]) != (
            "builtin.composite_effect",
            "1",
        ):
            raise ValueError("Physical receipt uses another call or monitor.")
        expectation = one(
            e
            for e in spec["state_expectations"]
            if e["object_id"] == c.source_uid
            and e["resource_id"] == c.resource_id
            and e["relation"] == "attached"
        )
        clause = one(
            e
            for e in spec["clauses"]
            if e["expectation_id"] == expectation["expectation_id"]
            and e.get("evidence_kind") == "constraint"
            and e.get("expected") is True
        )
        source = clause["source"]
        address = source["address"]
        if (
            source["provider_id"],
            source["revision"],
            address["control_part"],
            address["channel"],
        ) != ("builtin.control_part", "1", h.grasp_control_part, "constraint"):
            raise ValueError("Grip receipt does not use its native bound endpoint.")
        env_ids = spec["env_ids"]
        if len(set(env_ids)) != len(env_ids) or any(
            type(v) is not int for v in env_ids
        ):
            raise ValueError("Physical receipt environment mapping is invalid.")
        row = env_ids.index(c.env_id)
        samples = effect["evidence"]
        if any(
            s["timestamp"] != time
            or s["observation_revision"] != effect["observation_revision"]
            or s["env_ids"] != spec["env_ids"]
            or s["valid_mask"][row] is not True
            for s in samples.values()
        ):
            raise ValueError("Missing/invalid native guard evidence.")
        constraint = samples[clause["clause_id"]]
        query = one(
            q
            for q in h.native_grips
            if q["object_id"] == c.source_uid
            and q["control_part"] == h.grasp_control_part
            and q["timestamp"] == time
            and q["observation_revision"] == effect["observation_revision"]
            and q["expected"] is True
        )
        native = one(r for r in query["rows"] if r["env_id"] == c.env_id)
        raw = native["raw_contacts"]
        if any(
            len(pair) != 2 or any(type(v) is not int for v in pair)
            for pair in raw["pairs"]
        ):
            raise ValueError("Native contact actor IDs must be exact integers.")
        pairs = torch.as_tensor(raw["pairs"], dtype=torch.long).reshape(-1, 2)
        normals = torch.as_tensor(raw["normals"], dtype=torch.float64).reshape(-1, 3)
        impulses = torch.as_tensor(raw["impulses"], dtype=torch.float64)
        if (
            len(pairs) != len(normals)
            or impulses.shape != (len(pairs),)
            or query["dropped_contacts"] != 0
            or native["finite"] is not True
        ):
            raise ValueError("Native grip contact coverage is unavailable.")
        value, valid = classify_contacts(
            pairs, normals, impulses, target=h.object_actor_id, pads=h.pad_actor_ids
        )
        if (
            not valid
            or native["valid"] is not True
            or native["value"] is not value
            or constraint["values"][row] is not value
        ):
            raise ValueError("Native grip sample is unresolved or inconsistent.")
        decision = effect["decision"]
        if (
            type(decision["failure_mask"][row]) is not bool
            or type(decision["success_mask"][row]) is not bool
        ):
            raise ValueError("Physical monitor masks are not booleans.")
        lost |= decision["failure_mask"][row]
        selected.append((time, kind, decision["success_mask"][row], value))
        if lost:
            return True, {
                "contradicted_at": time,
                "guard_samples_observed": len(selected),
            }
    if not any(
        math.isclose(t, start, abs_tol=1e-8, rel_tol=0)
        and k == "phase_effect_gate"
        and s
        and v
        for t, k, s, v in selected
    ):
        raise ValueError("No fresh acquisition gate before Pour dispatch.")
    # The successful gate protects first dispatch. Subsequent fresh observations
    # guard the acknowledged command, including the seeded post-motion state.
    for step in range(c.start_step + 1, c.end_step + 1):
        one(
            s
            for s in selected
            if s[1] == "in_flight_guard"
            and math.isclose(s[0], step * c.step_dt, abs_tol=1e-8, rel_tol=0)
        )
    if not any(t >= end and k == "terminal" and s and v for t, k, s, v in selected):
        raise ValueError("No native terminal hold after the Pour window.")
    return lost, {
        "guard_samples": c.end_step - c.start_step,
        "acquisition_step": c.start_step,
        "guard_start_step": c.start_step + 1,
        "guard_end_step": c.end_step,
        "planned_at": c.planned_at,
        "attempt_generation": attempt.get("attempt_generation"),
        "native_actor_ids": actors,
    }


def evaluate_pour_pose_proxy(
    contract: PourProxyContract, window: NativePourWindow, hold: PourHoldEvidence
) -> dict[str, Any]:
    """Evaluate measured tilt/restore and held coverage, not return or fluid flow.

    The caller supplies the exact recorded motion boundary, including its seeded
    post-motion pose. Existing return/release/Park checks remain independent.
    Missing/ambiguous native evidence is unavailable rather than a false success.
    """
    result: dict[str, Any] = {
        "criterion": "e3_pose_proxy/v1",
        "fluid_status": "unmodeled",
        "status": "unavailable",
        "accepted": None,
    }
    try:
        c = contract
        if (
            not all(
                type(v) is str and v
                for v in (c.source_uid, c.receiver_uid, c.resource_id, c.invocation_id)
            )
            or c.source_uid == c.receiver_uid
        ):
            raise ValueError("Distinct exact source/receiver identities are required.")
        if (
            any(
                type(v) is not int or v < 0
                for v in (c.env_id, c.invocation_revision, c.start_step, c.end_step)
            )
            or c.end_step - c.start_step < 2
        ):
            raise ValueError("Invalid Pour identity or recorded step window.")
        if (window.source_uid, window.receiver_uid, window.env_id) != (
            c.source_uid,
            c.receiver_uid,
            c.env_id,
        ):
            raise ValueError("Native pose window belongs to another participant.")
        if window.steps != tuple(range(c.start_step, c.end_step + 1)) or any(
            type(s) is not int for s in window.steps
        ):
            raise ValueError(
                "Native source/receiver window must be contiguous and complete."
            )
        if any(
            type(v) not in (float, int) or not math.isfinite(v) or v <= 0
            for v in (
                c.step_dt,
                c.tilt_tolerance_rad,
                c.restore_tolerance_rad,
                c.position_tolerance_m,
            )
        ):
            raise ValueError("Explicit positive dt and proxy tolerances are required.")
        _clock(c.planned_at)
        count = len(window.steps)
        poses, receiver = _poses(window.source_poses, count), _poses(
            window.receiver_poses, count
        )
        entry, peak = _poses(c.entry_pose[None], 1), _poses(c.peak_pose[None], 1)
        _, separation = delta(entry, peak)
        if separation[0] <= c.tilt_tolerance_rad + c.restore_tolerance_rad:
            raise ValueError(
                "Entry/peak tolerance balls overlap; proxy cannot prove motion."
            )
        _, tilt_error = delta(poses, peak.expand_as(poses))
        _, restore_error = delta(poses, entry.expand_as(poses))
        if restore_error[0] > c.restore_tolerance_rad:
            raise ValueError("Measured entry differs from the declared native entry R.")
        pivot = torch.as_tensor(c.pivot_object, dtype=torch.float64)
        anchor = torch.as_tensor(c.receiver_anchor_local, dtype=torch.float64)
        if (
            pivot.shape != (3,)
            or anchor.shape != (3,)
            or not torch.isfinite(pivot).all()
            or not torch.isfinite(anchor).all()
        ):
            raise ValueError(
                "Explicit finite native-frame pivot/receiver anchor required."
            )
        measured = poses[:, :3, :3] @ pivot + poses[:, :3, 3]
        relative = (
            receiver[:, :3, :3].transpose(-1, -2)
            @ (measured - receiver[:, :3, 3])[..., None]
        )[..., 0]
        position_error = torch.linalg.vector_norm(relative - anchor, dim=-1)
        lost, coverage = _hold(c, hold)
        peaks = torch.where(tilt_error[1:-1] <= c.tilt_tolerance_rad)[0] + 1
        peak_index = int(peaks[0]) if len(peaks) else None
        restores = torch.where(restore_error <= c.restore_tolerance_rad)[0]
        restores = (
            restores[restores > peak_index] if peak_index is not None else restores[:0]
        )
        restored = len(restores) > 0 and restore_error[-1] <= c.restore_tolerance_rad
        accepted = (
            not lost
            and peak_index is not None
            and bool(restored)
            and bool((position_error <= c.position_tolerance_m).all())
        )
        reason = (
            "held_guard_contradicted"
            if lost
            else (
                "tilt_not_observed"
                if peak_index is None
                else (
                    "restore_not_observed"
                    if not restored
                    else (
                        "receiver_position_contradicted"
                        if not accepted
                        else "pose_proxy_satisfied"
                    )
                )
            )
        )
        result.update(
            status="satisfied" if accepted else "rejected",
            accepted=accepted,
            reason=reason,
            measurements={
                "identity": {
                    "source_uid": c.source_uid,
                    "receiver_uid": c.receiver_uid,
                    "resource_id": c.resource_id,
                    "invocation_id": c.invocation_id,
                    "invocation_revision": c.invocation_revision,
                    "env_id": c.env_id,
                    "start_step": c.start_step,
                    "end_step": c.end_step,
                    "step_dt": c.step_dt,
                },
                "peak_step": None if peak_index is None else window.steps[peak_index],
                "restore_step": (
                    None if not len(restores) else window.steps[int(restores[0])]
                ),
                "tilt_error_rad": tilt_error.tolist(),
                "restore_error_rad": restore_error.tolist(),
                "measured_rotation_from_entry_rad": restore_error.tolist(),
                "receiver_position_error_m": position_error.tolist(),
                "coverage": coverage,
                "tolerances": {
                    "tilt_rad": c.tilt_tolerance_rad,
                    "restore_rad": c.restore_tolerance_rad,
                    "position_m": c.position_tolerance_m,
                },
            },
        )
    except (KeyError, ValueError, TypeError, IndexError, RuntimeError) as error:
        result["reason"] = str(error) or type(error).__name__
    return result
