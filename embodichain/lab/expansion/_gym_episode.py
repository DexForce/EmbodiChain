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

"""Assemble measured numeric episodes and qualify their source evidence."""

from __future__ import annotations
import hashlib
import math
from pathlib import Path
import time
import numpy as np
import torch
from embodichain.data_pipeline.recording.media import _atomic_json
from .contracts import (
    ExpertEpisode,
    ValidationCheck,
    ValidationResult,
)


from ._gym_utils import _hash
from ._gym_context import GymHostContext

__all__: list[str] = []


class GymEpisodeAssembler:
    """Convert frozen cohort buffers into canonical accepted or rejected evidence."""

    def __init__(self, context: GymHostContext) -> None:
        self.context = context

    def assemble(self, assignments, collector, services, result, physics_after):
        ctx = self.context
        cfg, report, env = ctx.cfg, ctx.report, ctx.env
        names, limits, timings = ctx.names, ctx.limits, ctx.timings
        source_artifacts, profile, program = (
            ctx.source_artifacts,
            ctx.profile,
            ctx.program,
        )
        collision, physics_before = ctx.collision, ctx.physics_before
        records, spools = collector.records, collector.spools
        phase_started = time.monotonic()
        episodes = {}
        for result_row, row in enumerate(result.env_ids):
            identity, episode_id, commit_id, directory, record = assignments[row]
            record["program_segments"] = result.segments[result_row]
            record["steps"] = result.lengths[result_row]
            record["reason"] = result.reasons[result_row]
            if not result.accepted[result_row]:
                record["status"] = "rejected"
                measured = None if collision is None else collision.results.get(row)
                if measured is not None:
                    record["measured_collision"] = {
                        "status": measured.status,
                        "reason": measured.reason,
                    }
                if row in services and hasattr(services[row], "tracer"):
                    record["source_execution_reports"] = dict(
                        services[row].tracer.execution_reports
                    )
                partial = records[row]
                path = directory / "rejected.npz"
                arrays = {
                    "qpos": torch.stack(partial["qpos"]).numpy(),
                    "timestamps": np.arange(len(partial["qpos"])) * env.step_dt,
                }
                if partial["actions"]:
                    arrays["actions"] = torch.stack(partial["actions"]).numpy()
                arrays.update(
                    {
                        "object_" + str(index): torch.stack(values).numpy()
                        for index, values in enumerate(partial["objects"].values())
                    }
                )
                with path.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                record["partial_rollout"] = {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "object_uids": list(partial["objects"]),
                }
                _atomic_json(directory / "attempt.json", record)
                episodes[row] = None
                continue
            numeric = records[row]
            positions = torch.stack(numeric["qpos"])
            observations = {
                "joint_positions": positions[:, list(env.active_joint_ids)],
                "full_joint_positions": positions,
                "joint_velocities": torch.stack(numeric["qvel"])[
                    :, list(env.active_joint_ids)
                ],
                "joint_efforts": torch.stack(numeric["qf"])[
                    :, list(env.active_joint_ids)
                ],
            }
            observations.update(
                {
                    f"rigid/{uid}/state": torch.stack(values)
                    for uid, values in numeric["objects"].items()
                }
            )
            scene_keys = []
            for index, ((uid, field), values) in enumerate(numeric["scene"].items()):
                key = f"scene_state_{index}"
                observations[key] = torch.stack(values)
                scene_keys.append({"key": key, "entity_uid": uid, "field": field})
            metadata = {
                "attempt_sequence": ctx.requests[row].sequence,
                "job_instance_id": ctx.journal.job_instance_id,
                "recipe_id": ctx.requests[row].recipe.recipe_id,
                "attempt_number": ctx.requests[row].attempt_id,
                "recipe_spec": ctx.requests[row].recipe.to_dict(),
                "compatibility_key": ctx.compatibility.key,
                "expansion_source_artifacts": source_artifacts,
                "data_type": "sim",
                "runtime_versions": report["runtime_versions"],
                "observation_origin": "simulation",
                "visual_profile": cfg.visual_profile,
                "visual_profile_revision": report["input_files"].get(
                    "visual_profile_file"
                ),
                "recipe": record["recipe"],
                "scene_case": record["case"],
                "joint_names": names,
                "source_episode_uuid": _hash(("task_program", program.program_id)),
                "expert_action_spec": env.expert_action_spec.metadata(
                    step_dt=env.step_dt
                ),
                "initial_state": record["initial_state"],
                "scene_state_fields": scene_keys,
                "physics_backend": env.sim.physics_backend,
                "program_segments": result.segments[result_row],
                "expansion": (
                    result.expansion_records[result_row]
                    if result.expansion_records
                    else ()
                ),
                "segments": result.segments[result_row],
                "instruction": program.instruction,
                "instruction_source": "task_program",
            }
            physics_unchanged = all(
                torch.allclose(
                    physics_before[key][row],
                    physics_after[key][row],
                    atol=1e-6,
                    rtol=1e-6,
                )
                for key in physics_before
            )

            def parameter_value(value):
                if isinstance(value, list):
                    return [parameter_value(item) for item in value]
                return (
                    value if math.isfinite(value) else ("+inf" if value > 0 else "-inf")
                )

            metadata["physics_parameters"] = {
                key: parameter_value(value[row].detach().cpu().tolist())
                for key, value in physics_before.items()
            }
            metadata["physics_fingerprint"] = _hash(metadata["physics_parameters"])
            checks = list(ctx.preparation_checks[row].checks) + [
                ValidationCheck(
                    "physics_parameters",
                    "passed" if physics_unchanged else "failed",
                    "Measured episode parameter baseline",
                ),
                ValidationCheck(
                    "task_success",
                    "passed",
                    "Canonical Task Program and segment validators completed",
                ),
            ]
            if collision is not None:
                measured = collision.terminal.get(row)
                if measured is None:
                    checks.append(ValidationCheck("task.measured", "unavailable"))
                else:
                    checks.extend(measured.checks)
                checked = collision.results.get(row)
                checks.append(
                    ValidationCheck(
                        "path_collision",
                        (
                            "passed"
                            if checked is not None and checked.status == "passed"
                            else "failed"
                        ),
                        (
                            "Measured control-boundary collision checks"
                            if checked is None
                            else checked.reason
                        ),
                    )
                )
            if row in services:
                from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
                    Real2SimAcceptanceRejected,
                )

                try:
                    request = services[row].qualify_episode(
                        qpos=positions,
                        objects={
                            uid: torch.stack(values)
                            for uid, values in numeric["objects"].items()
                        },
                    )
                except Real2SimAcceptanceRejected as rejection:
                    rejected = ValidationResult(
                        tuple(checks) + rejection.validation.checks
                    )
                    record.update(
                        status="rejected",
                        reason="source_acceptance_failed",
                        measured_accepted=False,
                        validation=[
                            dict(
                                check_id=c.check_id,
                                status=c.status,
                                detail=c.detail,
                                metrics=dict(c.metrics),
                            )
                            for c in rejected.checks
                        ],
                    )
                    _atomic_json(directory / "attempt.json", record)
                    episodes[row] = None
                    continue
                # Source qualification keeps its legacy labels, while the host
                # owns executed action/state/schema and recipe identities.
                owned = set(metadata) - {"recipe"}
                metadata["motion_source_metadata"] = {
                    key: value
                    for key, value in request.metadata.items()
                    if key in owned
                }
                metadata.update(
                    {
                        key: value
                        for key, value in request.metadata.items()
                        if key not in owned
                    }
                )
                metadata["source_origin"] = request.metadata.get("origin")
                metadata.pop("source_episode_uuid", None)
                metadata["source_episode_uuids"] = request.source_episode_uuids
                checks.extend(request.validation.checks)
                if profile is not None:
                    checks.append(
                        ValidationCheck(
                            "path_collision",
                            "passed",
                            "Source-qualified planned and measured collision checks",
                        )
                    )
            if row in spools:
                metadata["media_manifest"] = str(spools[row].finish())
                metadata["media_manifest_sha256"] = hashlib.sha256(
                    Path(metadata["media_manifest"]).read_bytes()
                ).hexdigest()
            from embodichain.lab.sim.motion.expansion.coverage import (
                describe_trajectory,
            )

            descriptor = describe_trajectory(
                observations["joint_positions"],
                torch.arange(len(positions), dtype=torch.float64) * env.step_dt,
                limits,
            )
            metadata["coverage_descriptor"] = {
                "phase_ids": descriptor.phase_ids,
                "geometry": descriptor.geometry.tolist(),
                "timing": descriptor.timing.tolist(),
                "normalization": "joint_limits_arc_length",
            }
            metadata["expansion_recipe"] = record["recipe"]
            metadata["motion_origin"] = (
                "real2sim_prototype"
                if profile is not None
                else ("demo_skill_mixed" if row in services else "native_planning")
            )
            metadata["action_origin"] = "submitted_controller_command"
            metadata["state_origin"] = "measured_simulation"
            episode = ExpertEpisode(
                identity,
                observations,
                torch.stack(numeric["actions"]),
                torch.arange(len(positions), dtype=torch.float64) * env.step_dt,
                (
                    "joint_position_velocity"
                    if env.expert_action_spec.joint_command_mode == "position_velocity"
                    else "joint_position"
                ),
                ValidationResult(tuple(checks)),
                episode_id,
                commit_id,
                metadata=metadata,
            )
            episodes[row] = episode
            record["measured_accepted"] = episode.validation.accepted
            record["validation"] = [
                dict(
                    check_id=check.check_id,
                    status=check.status,
                    detail=check.detail,
                    metrics=dict(check.metrics),
                )
                for check in episode.validation.checks
            ]
            _atomic_json(directory / "attempt.json", record)
        timings["acceptance"] = (
            timings.get("acceptance", 0.0) + time.monotonic() - phase_started
        )
        return episodes
