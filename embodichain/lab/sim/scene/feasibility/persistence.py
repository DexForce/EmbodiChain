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

"""Closed, content-addressed JSON artifacts without executable deserialization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
import json
import numpy as np

from ..expansion import ScenePoseChange, SceneVariant
from ._values import atomic_json, digest, plain
from .cfg import SceneRobotFeasibilityCfg
from .contracts import (
    PoseSearchVariable,
    PhaseTarget,
    HeldObject,
    TaskPhase,
    TaskOperationMode,
    TaskFeasibilityRequest,
    RobotDeploymentCandidate,
    JointSceneCandidate,
    CandidateEvaluation,
    FeasibilityCheck,
)
from .search import (
    FeasibleRegion,
    FeasibilityAtlas,
    FeasibilityObservation,
    prepare_feasibility,
)

__all__ = ["request_from_mapping", "save_feasibility_atlas", "load_feasibility_atlas"]


def _closed(cls: type, value: Mapping[str, object]) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{cls.__name__} requires a mapping")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    return dict(value)


def request_from_mapping(value: Mapping[str, object]) -> TaskFeasibilityRequest:
    """Decode a provider-independent request through closed production contracts.

    Args:
        value: JSON/YAML task, scene, mounting and phase declarations.
    Returns:
        Validated immutable request. No dotted import or model client is loaded.
    """
    data = _closed(TaskFeasibilityRequest, value)
    data["variables"] = tuple(
        PoseSearchVariable(**_closed(PoseSearchVariable, v)) for v in data["variables"]
    )
    modes = []
    for mode in data["modes"]:
        m = _closed(TaskOperationMode, mode)
        phases = []
        for phase in m["phases"]:
            p = _closed(TaskPhase, phase)
            for key, cls in (
                ("targets", PhaseTarget),
                ("held_objects", HeldObject),
                ("resulting_poses", ScenePoseChange),
            ):
                p[key] = tuple(cls(**_closed(cls, v)) for v in p.get(key, ()))
            phases.append(TaskPhase(**p))
        m["phases"] = tuple(phases)
        modes.append(TaskOperationMode(**m))
    data["modes"] = tuple(modes)
    return TaskFeasibilityRequest(**data)


def save_feasibility_atlas(atlas: FeasibilityAtlas, path: str | Path) -> None:
    """Atomically publish a frozen, content-addressed evidence snapshot.

    Args:
        atlas: Complete owned region and observation artifact.
        path: Destination JSON path.
    """
    data = plain(atlas)
    atomic_json(
        Path(path),
        {
            "schema": "scene-robot-feasibility/v1",
            "content_id": digest(data),
            "atlas": data,
        },
    )


def load_feasibility_atlas(path: str | Path) -> FeasibilityAtlas:
    """Validate source identity, every candidate, region and statistical receipt.

    Args:
        path: Saved JSON artifact; only the explicit v1 schema is accepted.
    Returns:
        Owned atlas, with proposals reconstructed from its fixed joint domain.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        set(data) != {"schema", "content_id", "atlas"}
        or data["schema"] != "scene-robot-feasibility/v1"
        or digest(data["atlas"]) != data["content_id"]
    ):
        raise ValueError("Invalid or changed feasibility artifact")
    values = _closed(FeasibilityAtlas, data["atlas"])
    values["request"] = request_from_mapping(values["request"])
    values["cfg"] = SceneRobotFeasibilityCfg.from_mapping(values["cfg"])
    cpu_settings = SceneRobotFeasibilityCfg.from_mapping(
        {**plain(values["cfg"]), "device": "cpu"}
    )
    prepared = prepare_feasibility(values["request"], cpu_settings)
    observations = []
    for item in values["observations"]:
        obs = _closed(FeasibilityObservation, item)
        raw = _closed(JointSceneCandidate, obs["candidate"])
        scene = raw["scene"]
        expected = prepared.candidate(
            raw["parameters"],
            mode_id=raw["mode_id"],
            ordinal=scene["ordinal"],
            seed=scene["seed"],
        )
        scene_values = _closed(SceneVariant, scene)
        scene_values["pose_changes"] = tuple(
            ScenePoseChange(**_closed(ScenePoseChange, v))
            for v in scene_values["pose_changes"]
        )
        candidate = JointSceneCandidate(
            raw["request_id"],
            SceneVariant(**scene_values),
            RobotDeploymentCandidate(
                **_closed(RobotDeploymentCandidate, raw["deployment"])
            ),
            raw["mode_id"],
            raw["parameters"],
        )
        normalized = plain(candidate)
        expected_values = plain(expected)
        # Reading an atlas must never require its originating CUDA device. Keep
        # saved poses/identities verbatim while checking canonical CPU geometry
        # within the declared float64 CPU/CUDA materialization tolerance.
        actual_changes = normalized["scene"]["pose_changes"]
        expected_changes = expected_values["scene"]["pose_changes"]
        if len(actual_changes) != len(expected_changes):
            raise ValueError("Candidate pose set disagrees with its request")
        for actual, target in zip(actual_changes, expected_changes, strict=True):
            if actual["entity_uid"] != target["entity_uid"] or not np.allclose(
                actual["arena_pose"], target["arena_pose"], atol=1e-12, rtol=0
            ):
                raise ValueError("Candidate geometry disagrees with its request")
            actual["arena_pose"] = target["arena_pose"]
        actual_roots = normalized["deployment"]["poses"]
        target_roots = expected_values["deployment"]["poses"]
        if actual_roots.keys() != target_roots.keys() or any(
            not np.allclose(actual_roots[k], v, atol=1e-12, rtol=0)
            for k, v in target_roots.items()
        ):
            raise ValueError("Candidate installation disagrees with its request")
        normalized["deployment"]["poses"] = target_roots
        if normalized != expected_values:
            raise ValueError("Candidate identity disagrees with its request")
        evidence = _closed(CandidateEvaluation, obs["evaluation"])
        evidence["checks"] = tuple(
            FeasibilityCheck(**_closed(FeasibilityCheck, c)) for c in evidence["checks"]
        )
        observations.append(
            FeasibilityObservation(
                candidate,
                CandidateEvaluation(**evidence),
                obs.get("region_id"),
                obs.get("completed", True),
            )
        )
    values["observations"] = tuple(observations)
    regions = []
    for item in values["regions"]:
        region = FeasibleRegion(**_closed(FeasibleRegion, item))
        expected_id = digest(
            (values["request"].request_id, region.mode_id, (region.lower, region.upper))
        )
        if region.region_id != expected_id:
            raise ValueError("Region identity disagrees with its joint bounds")
        regions.append(region)
    values["regions"] = tuple(regions)
    return FeasibilityAtlas(**values)
