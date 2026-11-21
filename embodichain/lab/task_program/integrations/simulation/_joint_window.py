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

"""Bounded raw joint-window records; no policy judgment or simulator access."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
from embodichain.lab.sim.evidence import PhysicalEvidenceBatch
from typing import Any
import torch

__all__: list[str] = []


class JointWindowRecord:
    """Own bounded samples independently of an execution generator's lifetime."""

    def __init__(
        self,
        capacity: int,
        *,
        initial: torch.Tensor,
        condition: dict[str, object],
        env_ids: torch.Tensor,
        native_uid: str,
        segment_id: str,
        policy_source_path: list[str | int],
        observation_generation: int,
        clock_start: tuple[int, float] | None,
        step_dt: float,
    ) -> None:
        self.initial = initial.detach().clone()
        self.condition = deepcopy(condition)
        self.env_ids = env_ids.clone()
        self.native_uid = native_uid
        self.segment_id = segment_id
        self.policy_source_path = list(policy_source_path)
        self.generation = observation_generation
        self.clock_start = clock_start
        self.step_dt = step_dt
        self.values: deque[torch.Tensor] = deque(maxlen=capacity)
        self.steps: deque[int] = deque(maxlen=capacity)
        self.timestamps: deque[float] = deque(maxlen=capacity)
        self.total = 0

    def append(
        self, value: torch.Tensor, step: int | None, timestamp: float | None
    ) -> None:
        self.values.append(value.detach().clone())
        if step is not None:
            self.steps.append(step)
            self.timestamps.append(timestamp)
        self.total += 1

    def snapshot(self) -> dict[str, Any]:
        if not self.values:
            return dict(
                sample_count=0,
                retained_sample_count=0,
                dropped_samples=0,
                record_complete=False,
                steps=[],
                timestamps=[],
                values=[],
                valid=[],
            )
        values = torch.stack(tuple(self.values), dim=1)
        valid = torch.isfinite(values)
        encoded = values.masked_fill(~valid, 0).detach().cpu().tolist()
        masks = valid.detach().cpu().tolist()
        return dict(
            sample_count=self.total,
            retained_sample_count=len(self.values),
            dropped_samples=self.total - len(self.values),
            record_complete=len(self.values) == self.total and bool(self.steps),
            steps=list(self.steps),
            timestamps=list(self.timestamps),
            values=[
                [
                    value if available else None
                    for value, available in zip(row, mask, strict=True)
                ]
                for row, mask in zip(encoded, masks, strict=True)
            ],
            valid=masks,
            value_dtype=str(values.dtype),
            value_shape=list(values.shape),
            acquisition_errors=[
                [
                    None if valid else "Non-finite articulation joint observation."
                    for valid in row
                ]
                for row in masks
            ],
        )

    def finish(
        self,
        last: torch.Tensor,
        *,
        active_mask: torch.Tensor,
        unavailable: torch.Tensor,
        accepted: torch.Tensor,
        completed: bool,
        termination: str,
        invalid_observation_policy: str,
        interruption_reason: str | None,
    ) -> dict[str, Any]:
        """Serialize retained facts and execution outcome as separate fields."""
        cfg = self.condition
        record = dict(
            joint=cfg["joint"],
            target=cfg["target"],
            observed=[float(v) if math.isfinite(float(v)) else None for v in last],
            tolerance=cfg["tolerance"],
            stable_seconds=cfg["duration"],
            accepted=accepted.tolist(),
            evidence="joint_position_and_stability",
            contact_qualification="not_provided",
        )
        if (
            self.clock_start is not None
            or not completed
            or invalid_observation_policy != "abort_batch"
        ):
            statuses = [
                (
                    "inactive"
                    if not active
                    else (
                        "unavailable"
                        if missing
                        else (
                            "satisfied"
                            if success
                            else (
                                "not_satisfied"
                                if termination == "timed_out"
                                else "pending"
                            )
                        )
                    )
                )
                for active, missing, success in zip(
                    active_mask.tolist(),
                    unavailable.tolist(),
                    accepted.tolist(),
                    strict=True,
                )
            ]
            window = self.snapshot()
            initial_valid = torch.isfinite(self.initial).tolist()
            window.update(
                schema_version="simulation.joint_stability_window/v1",
                entity=cfg["entity"],
                joint=cfg["joint"],
                native_uid=self.native_uid,
                segment_id=self.segment_id,
                policy_source_path=self.policy_source_path,
                env_ids=self.env_ids.tolist(),
                value_unit="joint_runtime_units",
                monitor_initial_values=[
                    float(value) if valid else None
                    for value, valid in zip(self.initial, initial_valid, strict=True)
                ],
                monitor_initial_valid=initial_valid,
                criterion=deepcopy(cfg),
                termination=termination,
                condition_status=statuses,
                active_mask=active_mask.tolist(),
                unavailable_mask=unavailable.tolist(),
                execution_complete=completed,
                invalid_observation_policy=invalid_observation_policy,
                observation_generation=self.generation,
                interruption_reason=interruption_reason,
                step_dt=self.step_dt,
            )
            record["joint_window"] = window
        if self.clock_start is not None and self.timestamps:
            finite = torch.isfinite(last)
            raw = PhysicalEvidenceBatch(
                "articulation.joint_position",
                last,
                finite,
                tuple(
                    None if value else "Non-finite articulation joint observation."
                    for value in finite.tolist()
                ),
                self.timestamps[-1],
                self.env_ids,
                self.total,
            )
            record["joint_observation"] = dict(
                evidence_id=raw.evidence_id,
                values=record["observed"],
                valid=raw.valid.tolist(),
                acquisition_errors=list(raw.acquisition_errors),
                timestamp=raw.timestamp,
                env_ids=raw.env_ids.tolist(),
                observation_revision=raw.observation_revision,
                revision_scope="policy_window_qpos_acquisition_sequence",
                clock="environment_step",
                start_step=self.clock_start[0],
                end_step=self.steps[-1],
                started_at=self.clock_start[1],
                step_dt=self.step_dt,
            )
        return record
