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

"""Occurrence-bound source preparation and measured qualification for native hosts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from collections.abc import Mapping
import hashlib
import json

from .contracts import ValidationCheck, ValidationResult
from ._gym_utils import _hash

__all__: list[str] = []


@dataclass(frozen=True)
class _SourceQualification:
    metadata: Mapping[str, object]
    validation: ValidationResult
    source_episode_uuids: tuple[str, ...]


class _EpisodeSource(Protocol):
    """Non-owning row adapter; stepping and persistence remain with the host."""

    bridge: object

    def qualify_episode(self, *, qpos, objects) -> _SourceQualification:
        """Validate measured source evidence after canonical execution."""


class _OccurrenceSourceRow:
    def __init__(
        self,
        env,
        row,
        program,
        *,
        expansion_profile=None,
        candidate_index=0,
        duration_scale=1.0,
        velocity_limits=(),
        acceleration_limits=(),
        prototype_profile=None,
    ):
        from embodichain.lab.task_program.integrations.demo_skill import (
            _ConfiguredDemoSkillSource,
        )

        self.adapter = env.task_program_adapter.for_env_ids((row,))
        self.declarations = self.adapter.configured_nominal_sources
        self.program = self.adapter.compile(env.cfg.task_program)
        if self.program.program_id != program.program_id:
            raise ValueError("source row program differs from the cohort")
        self.prototype_source = None
        self.prototype_validator = None
        from embodichain.lab.task_program.integrations.nominal import (
            _ConfiguredPrototypeSource,
        )

        prototype_calls = tuple(
            item.call_index
            for item in self.declarations
            if isinstance(item, _ConfiguredPrototypeSource)
        )
        guard = None
        if prototype_calls:
            if prototype_profile is None:
                raise ValueError(
                    "mixed prototype calls require prototype_validation_profile"
                )
            from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
                Real2SimTabletopSource,
            )
            from embodichain.lab.task_program.integrations.simulation.real2sim_collision import (
                TabletopPrototypeCollisionValidator,
                TabletopCollisionGuardFactory,
            )

            source = Real2SimTabletopSource(
                env, prototype_profile, env_id=row, call_indices=prototype_calls
            )
            self.prototype_source = source
            self.prototype_validator = TabletopPrototypeCollisionValidator(
                robot=env.robot,
                checker=source.checker,
                prototypes=tuple(source.unique_prototypes.values()),
                object_uid=prototype_profile.object_entity_id,
                target_uid=prototype_profile.target_entity_id,
                ground_uid="ground",
                finger_links=prototype_profile.finger_links,
                finger_joints=prototype_profile.finger_joints,
                closing_directions=prototype_profile.closing_directions,
                gripper_contact=prototype_profile.gripper_contact,
                base_link=prototype_profile.base_link,
                contact_depth_m=prototype_profile.contact_depth_m,
                base_ground_depth_m=prototype_profile.base_ground_depth_m,
                env_id=row,
            )
            guard = TabletopCollisionGuardFactory(
                self.prototype_validator,
                call_indices=prototype_calls,
                velocity_limits=prototype_profile.velocity_limits,
                acceleration_limits=prototype_profile.acceleration_limits,
            )
        self.tracer = self.adapter.create_traced_configured_sources(self.program, guard)
        transform = None
        if duration_scale != 1:
            from embodichain.lab.task_program.integrations.expansion import (
                TaskProgramMappedRetimeFactory,
            )

            transform = TaskProgramMappedRetimeFactory(
                duration_scale=duration_scale,
                velocity_limits=velocity_limits,
                acceleration_limits=acceleration_limits,
                call_indices=tuple(
                    item.call_index
                    for item in self.declarations
                    if isinstance(item, _ConfiguredDemoSkillSource)
                ),
                diagnostic_unchecked=True,
            )
        bridge_profile = expansion_profile
        bridge_candidate = candidate_index
        if guard is not None and expansion_profile is not None:
            from embodichain.lab.task_program.integrations.expansion import (
                CombinedTaskProgramExpansionFactory,
            )

            transform = CombinedTaskProgramExpansionFactory(
                expansion_profile,
                candidate_index=candidate_index,
                program_id=self.program.program_id,
                integration_id=self.program.integration.scene_registry,
                robot_profile_id=self.program.integration.robot_profile,
            )
            bridge_profile = None
            bridge_candidate = 0
        if guard is not None:
            guard.delegate = transform
            transform = guard
        self.bridge = self.adapter.create_bridge(
            self.program,
            nominal_source_factory=self.tracer,
            expansion_profile=bridge_profile,
            candidate_index=bridge_candidate,
            plan_transform_factory=transform,
        )
        self.bridge.enable_execution_trace()
        self.demo_calls = {
            item.call_index: item
            for item in self.declarations
            if isinstance(item, _ConfiguredDemoSkillSource)
        }
        if not self.demo_calls and not prototype_calls:
            raise ValueError(
                "occurrence source host requires configured motion sources"
            )

    def qualify_episode(self, *, qpos, objects) -> _SourceQualification:
        from embodichain.data_pipeline.real2sim import load_demo_skill

        del objects
        trace = self.bridge.execution_trace
        reports = self.tracer.execution_reports
        checks = [
            ValidationCheck(
                "source.action_state_trace",
                "passed" if len(trace) == len(qpos) - 1 else "failed",
            )
        ]
        metadata, artifacts, source_ids = {}, {}, []
        for index, declaration in self.demo_calls.items():
            declaration.validate_source()
            artifact = load_demo_skill(declaration.asset_manifest)
            report = reports.get(index)
            valid = (
                report is not None
                and report.get("asset_sha256") == artifact.content_sha256
                and report.get("terminal_passed") is True
                and all(
                    status == "passed" for status in report.get("checks", {}).values()
                )
            )
            checks.append(
                ValidationCheck(
                    f"demo_skill.call.{index}",
                    "passed" if valid else "failed",
                    "Measured event and terminal predicates",
                )
            )
            metadata[str(index)] = report
            root = Path(declaration.asset_manifest).parent
            manifest = json.loads(Path(declaration.asset_manifest).read_text())
            source = root / manifest["source_manifest"]
            source_manifest = json.loads(source.read_text())
            artifacts[str(index)] = {
                name: {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for name, path in (
                    ("asset_manifest", Path(declaration.asset_manifest)),
                    ("source_manifest", source),
                    ("source_arrays", root / source_manifest["array_file"]),
                )
            }
            if "original_recording" in manifest:
                path = root / manifest["original_recording"]["file"]
                artifacts[str(index)]["original_recording"] = {
                    "path": str(path.resolve()),
                    "sha256": manifest["original_recording"]["sha256"],
                }
            if "reference_file" in manifest:
                path = root / manifest["reference_file"]
                artifacts[str(index)]["nominal_reference"] = {
                    "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            source_ids.append(
                _hash(
                    (
                        "real2sim",
                        artifact.demonstration.source_id,
                        artifact.demonstration.source_revision,
                    )
                )
            )
        prototype_artifacts = {}
        if self.prototype_source is not None:
            source = self.prototype_source
            for declaration in source.declarations:
                path = Path(declaration.prototype_manifest)
                payload = json.loads(path.read_text())
                source_path = path.parent / payload["source_manifest"]
                source_payload = json.loads(source_path.read_text())
                references = {
                    "prototype_manifest": path,
                    "normalized_arrays": path.parent / payload["normalized_file"],
                    "source_manifest": source_path,
                    "source_arrays": source_path.parent / source_payload["array_file"],
                }
                prototype_artifacts[str(declaration.call_index)] = {
                    name: {
                        "path": str(value.resolve()),
                        "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
                    }
                    for name, value in references.items()
                }
                prototype_artifacts[str(declaration.call_index)].update(
                    source.profile.source_artifacts.get(declaration.call_index, {})
                )
            geometry_reports = self.prototype_validator.reports
            valid = len(geometry_reports) >= len(source.declarations) and all(
                report.accepted for _, report in geometry_reports
            )
            checks.append(
                ValidationCheck(
                    "prototype.planned_geometry", "passed" if valid else "failed"
                )
            )
            for unit in source.prototypes.values():
                source_ids.append(
                    _hash(
                        (
                            "real2sim",
                            unit.unit.source.source_id,
                            unit.unit.source.source_revision,
                        )
                    )
                )
        source_origins = [
            report.get("origin") for report in reports.values() if report is not None
        ]
        if self.prototype_source is not None:
            source_origins.extend(
                unit.unit.source.origin
                for unit in self.prototype_source.prototypes.values()
            )
        return _SourceQualification(
            {
                "demo_skill_execution": metadata,
                "demo_skill_source_artifacts": artifacts,
                "real2sim_source_artifacts": prototype_artifacts,
                "execution_trace": trace,
                "origin": (
                    "measured"
                    if source_origins
                    and all(origin == "measured" for origin in source_origins)
                    else "synthetic_reference"
                ),
                "source_origins": sorted(set(source_origins)),
                "observation_independent_expert": True,
            },
            ValidationResult(tuple(checks)),
            tuple(sorted(set(source_ids))),
        )
