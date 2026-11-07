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

"""Fixed-scene planning, trajectory playback artifacts and sample lifecycle."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

from embodichain.lab.sim.atomic_actions import (
    ActionPlan,
    CompiledTrajectory,
    TimedTrajectory,
)
from embodichain.lab.sim.atomic_actions._plan_expansion import (
    _PlanExpansionSpec,
    _expand_plan,
)
from embodichain.lab.sim.motion.expansion import (
    SceneCase,
    SourceContext,
    TrajectoryExpansionCfg,
)
from .defaults import AtomicSkillRunCfg

if TYPE_CHECKING:
    from embodichain.lab.sim.motion.planners.utils import PlanResult
    from embodichain.lab.sim.atomic_actions import (
        ActionInvocation,
        AtomicActionEngine,
        PlanningContext,
        ResolvedActionRequest,
    )

__all__: list[str] = []


def _row_sample(value: Any, row: int, batch_size: int) -> Any:
    if isinstance(value, dict):
        return {key: _row_sample(item, row, batch_size) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        selected = value[row] if value.ndim and len(value) == batch_size else value
        return selected.detach().cpu().tolist()
    if isinstance(value, (tuple, list)):
        if len(value) == batch_size:
            return _row_sample(value[row], 0, -1)
        return [_row_sample(item, row, batch_size) for item in value]
    return value


def _sample_geometry(value: Any, path: str = "") -> dict[str, torch.Tensor]:
    """Extract actual selected poses in their declared reference frames."""
    result = {}
    if not isinstance(value, dict):
        return result
    if "selected_poses" in value:
        pose = torch.as_tensor(value["selected_poses"], dtype=torch.float64)
        if pose.shape != (4, 4):
            raise ValueError(
                "row-local affordance metadata must contain a 4x4 selected pose"
            )
        reference = value.get("reference_poses")
        if reference is not None:
            pose = torch.linalg.inv(torch.as_tensor(reference, dtype=pose.dtype)) @ pose
        result[path or "pose"] = pose
    for name, child in value.items():
        if isinstance(child, dict):
            result.update(_sample_geometry(child, f"{path}/{name}"))
    return result


def _same_geometry(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> bool:
    if first.keys() != second.keys():
        return False
    return all(
        torch.allclose(first[key][:3, 3], second[key][:3, 3], atol=1e-4, rtol=0)
        and torch.allclose(first[key][:3, :3], second[key][:3, :3], atol=1e-3, rtol=0)
        for key in first
    )


@dataclass(frozen=True)
class _CompiledSample:
    reference: TimedTrajectory
    stages: tuple[dict[str, Any], ...]
    grasp_joint_indices: tuple[int, ...]


class AtomicSkillRun:
    """Own one finite parallel sample run without Gym or dataset collection."""

    def __init__(self, cfg: AtomicSkillRunCfg) -> None:
        self.cfg = cfg
        self.replays: list[dict[str, Any]] = []
        self._compiled: dict[int, _CompiledSample] = {}
        self._stage_index = 0
        self._trajectory_changed = [False] * cfg.variants
        self._affordance_geometry: list[dict[str, torch.Tensor]] = [
            {} for _ in range(cfg.variants)
        ]

    def compile(
        self,
        engine: AtomicActionEngine,
        invocations: Iterable[ActionInvocation],
        context: PlanningContext,
    ) -> CompiledTrajectory:
        """Expand complete plans before concatenation, retaining their references."""
        originals: list[ActionPlan] = []
        stages: list[dict[str, Any]] = []
        grasp_joint_indices: set[int] = set()

        def transform(
            request: ResolvedActionRequest, current: PlanningContext, plan: ActionPlan
        ) -> ActionPlan:
            stage_index = self._stage_index
            self._stage_index += 1
            originals.append(plan.snapshot())
            grasp_joint_indices.update(
                joint_id
                for endpoint in request.binding.endpoints
                if endpoint.endpoint_id == "grasp"
                for joint_id in endpoint.joint_ids
            )
            specs = []
            for row in range(current.batch_size):
                variant_id = int(current.env_ids[row])
                digest = hashlib.sha256(
                    repr(
                        (self.cfg.seed, self.cfg.skill, stage_index, variant_id)
                    ).encode()
                ).digest()
                seed = int.from_bytes(digest[:8], "little") % (2**63)
                requested = int(variant_id > 0 and self.cfg.trajectory_enabled)
                operator = ("joint_residual", "via_points")[(variant_id - 1) % 2]
                cfg = TrajectoryExpansionCfg.from_mapping(
                    {
                        "augmentation": {
                            "seed": seed,
                            "factors": {
                                "spatial": {
                                    "enabled": bool(requested),
                                    "method": [operator],
                                    "joint_offset_scale": 0.005,
                                    "via_count": 3,
                                },
                                "ik": {"enabled": False},
                                "timing": {"enabled": False},
                            },
                            "coverage": {"joint_dedup_normalized_tol": 0.002},
                        },
                        "affordance": {"enabled": False},
                        "scheduling": {"candidate_budget": 2},
                        "execution": {"max_inflight": 1},
                        "observation": {"enabled": False, "profiles": []},
                    }
                )
                case = SceneCase(
                    f"atomic_action:{self.cfg.skill}",
                    "fixed_initial",
                    "fixed_scene_v2",
                    self.cfg.skill,
                    self.cfg.robot,
                )
                source = SourceContext(
                    f"atomic_action:{self.cfg.skill}",
                    "sample_v2",
                    f"{self.cfg.skill}:stage:{stage_index}:variant:{variant_id}",
                    case,
                    current.require_control_dt(),
                )
                specs.append(
                    _PlanExpansionSpec(
                        source,
                        cfg,
                        requested,
                        1 + requested,
                        {
                            "variant_id": variant_id,
                            "affordance_branch": (
                                variant_id if self.cfg.affordance_enabled else 0
                            ),
                        },
                    )
                )
            rebuilt, results = _expand_plan(engine, request, current, plan, specs)
            rows = []
            for result in results:
                record = dict(result.metadata)
                record["affordance_samples"] = _row_sample(
                    plan.diagnostics.metadata.get("affordance_sample", {}),
                    result.row_index,
                    current.batch_size,
                )
                rows.append(record)
                variant_id = int(record["variant_id"])
                self._trajectory_changed[variant_id] |= record["operator"] != "none"
                geometry = _sample_geometry(record["affordance_samples"])
                self._affordance_geometry[variant_id].update(
                    {
                        f"stage:{stage_index}:{name}": pose
                        for name, pose in geometry.items()
                    }
                )
            stages.append(
                {"stage_index": stage_index, "skill_id": plan.skill_id, "rows": rows}
            )
            return rebuilt

        compiled = engine.compile(invocations, context, plan_transform=transform)
        reference = TimedTrajectory.concatenate(
            [plan.joint_trajectory for plan in originals],
            empty_like=context,
        )
        self._compiled[id(compiled)] = _CompiledSample(
            reference, tuple(stages), tuple(sorted(grasp_joint_indices))
        )
        return compiled

    def begin_replay(self, compiled: CompiledTrajectory, prefix: str) -> dict[str, Any]:
        """Persist the planned reference and expanded execution before playback."""
        self.validate_variants()
        sample = self._compiled.pop(id(compiled))
        directory = Path(self.cfg.output_dir)
        stem = f"{len(self.replays):02d}_{prefix}"
        record = {
            "segment": prefix,
            "status": "planned",
            "measured_task_success": None,
            "stages": list(sample.stages),
            "grasp_joint_indices": list(sample.grasp_joint_indices),
            "trajectory_path": None,
            "video_path": (
                str(directory / f"{stem}.mp4") if self.cfg.save_video else None
            ),
            "record_path": str(directory / f"{stem}.json"),
        }
        if self.cfg.save_trajectories:
            path = directory / f"{stem}.pt"
            torch.save(
                {
                    "schema_version": 2,
                    "positions": compiled.trajectory.positions.detach().cpu(),
                    "reference_positions": sample.reference.positions.detach().cpu(),
                    "dt": compiled.trajectory.dt.detach().cpu(),
                    "reference_dt": sample.reference.dt.detach().cpu(),
                    "env_ids": compiled.trajectory.env_ids.detach().cpu(),
                    "stages": sample.stages,
                },
                path,
            )
            record["trajectory_path"] = str(path)
        self.replays.append(record)
        self.write_replay(record)
        return record

    def write_replay(self, record: dict[str, Any]) -> None:
        Path(record["record_path"]).write_text(json.dumps(record, indent=2) + "\n")

    def finish_replay(
        self, record: dict[str, Any], executed: PlanResult, holds: dict[int, float]
    ) -> None:
        """Save the actual command timeline, including explicitly recorded holds."""
        record.update(
            status="executed",
            executed_duration=float(executed.dt.sum(dim=1).max()),
            hold_durations={str(index): duration for index, duration in holds.items()},
        )
        if record["trajectory_path"] is not None:
            path = Path(record["trajectory_path"])
            payload = torch.load(path, weights_only=True)
            payload["planned_positions"] = payload["positions"]
            payload["planned_dt"] = payload["dt"]
            payload.update(
                positions=executed.positions.detach().cpu(),
                velocities=executed.velocities.detach().cpu(),
                dt=executed.dt.detach().cpu(),
                hold_durations=record["hold_durations"],
            )
            torch.save(payload, path)
        self.write_replay(record)

    def validate_variants(self) -> None:
        """Refuse missing factors instead of counting repeated nominal executions."""
        for variant_id in range(1, self.cfg.variants):
            if self.cfg.trajectory_enabled and not self._trajectory_changed[variant_id]:
                raise ValueError(
                    f"variant {variant_id} has no accepted trajectory expansion"
                )
            if self.cfg.affordance_enabled:
                for previous in range(variant_id):
                    if _same_geometry(
                        self._affordance_geometry[variant_id],
                        self._affordance_geometry[previous],
                    ):
                        raise ValueError(
                            f"affordance variant {variant_id} repeats geometry of variant {previous}; "
                            "not enough distinct feasible candidates"
                        )
            elif not _same_geometry(
                self._affordance_geometry[variant_id], self._affordance_geometry[0]
            ):
                raise ValueError(
                    f"trajectory-only variant {variant_id} changed the reference affordance"
                )
