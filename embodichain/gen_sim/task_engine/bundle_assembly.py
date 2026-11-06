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

"""Finalize owner declarations into a preflighted atomic Task Program bundle."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
import yaml
from embodichain.gen_sim.task_spec import TaskSpec
from .bundle_spec import bundle_task_spec_payload
from embodichain.lab.task_program import TaskProgramIntegrationCfg
from embodichain.lab.task_program.integrations import (
    compile_configured_task_program,
    load_configured_task_program_deployment,
    task_program_fingerprint,
)
from .semantic_graph import SemanticTaskGraph, validate_semantic_task_graph
from .bundle_deployment import prepare_configured_bundle
from .bundle_io import BundleSnapshot, write_bundle_atomic
from .bundle_inputs import SOURCE_INPUT_PROVENANCE_FILE

__all__ = [
    "TaskProgramBundlePaths",
    "prepare_task_program_bundle",
    "publish_task_program_bundle",
]


@dataclass(frozen=True, slots=True)
class TaskProgramBundlePaths:
    """Canonical declaration paths, separate from run receipts and certificates."""

    root: Path
    deployment: Path
    program: Path
    integration: Path
    scene: Path
    embodiment: Path
    execution_policy: Path
    semantic_task_graph: Path
    integration_fingerprint: Path

    @property
    def source_input_provenance(self) -> Path:
        """Return the sealed source-owner path/hash inventory artifact."""
        return self.root / SOURCE_INPUT_PROVENANCE_FILE

    @classmethod
    def at(cls, root: str | Path) -> TaskProgramBundlePaths:
        """Normalize a fresh destination without opening declaration files.

        Args:
            root: Intended bundle directory.

        Returns:
            Existing public bundle path layout with no runtime artifacts.
        """
        root = Path(root).expanduser().resolve()
        return cls(
            root,
            root / "task_program_deployment.yaml",
            root / "task_program/program.yaml",
            root / "task_program/integration.yaml",
            root / "components/scene.yaml",
            root / "components/embodiment.yaml",
            root / "components/execution_policy.yaml",
            root / "semantic_task_graph.json",
            root / "integration_fingerprint.json",
        )


def _json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def prepare_task_program_bundle(
    graph: Mapping[str, Any],
    task_spec: TaskSpec | Mapping[str, Any] | None,
    *,
    program: Mapping[str, Any],
    integration: Mapping[str, Any],
    scene: Mapping[str, Any],
    embodiment: Mapping[str, Any],
    execution_policy: Mapping[str, Any],
    deployment: Mapping[str, Any],
    invocation_presets: Mapping[str, str],
    external_input_digests: Mapping[str, str],
    implementation_revisions: Mapping[str, str],
    base_dir: str | Path,
    artifact_declarations: Mapping[str, Mapping[str, Any]] | None = None,
    source_input_provenance: Mapping[str, Any] | None = None,
) -> tuple[SemanticTaskGraph, BundleSnapshot]:
    """Compile identities, bind named presets, preflight and seal pure declarations.

    Args:
        graph: Semantic graph before final integration identity is known.
        task_spec: Independent immutable normative specification.
        program: Owner-resolved program declaration from semantic segment assembly.
        integration: Task-owned scene/profile/runtime service declaration.
        scene: Physical scene component; no Task Engine/runtime fields are added.
        embodiment: Selected robot/sensor component with its explicit skill profile.
        execution_policy: Selected motion/recovery/runner policy component.
        deployment: Runnable declaration with canonical component selections.
        invocation_presets: Exact segment-name to named policy selection. Every
            segment must be explicit and contain one serial call; duplicates and
            unused names reject. Values never depend on an analysis cache index.
        external_input_digests: Independently acquired current asset/input hashes.
        implementation_revisions: Revisions of all selected implementation owners.
        base_dir: Source declaration owner directory used for resource resolution.
        artifact_declarations: Optional pure JSON provenance artifacts. Canonical
            runtime files cannot be replaced by these supplemental declarations.

    Returns:
        Independently owned fingerprint-bound graph and sealed declaration bytes.
        No files, simulator, IK query, skill object or live service is created.
        Existing source input and implementation identities must be supplied by
        their owners; old runtime envelopes/constraints are not converted.

    Raises:
        ValueError: Any declaration, physical mapping, occurrence, service or
            semantic identity fails static preflight.
    """
    selected_graph = validate_semantic_task_graph(graph)
    spec = bundle_task_spec_payload(task_spec)
    owned = {
        name: deepcopy(dict(value))
        for name, value in (
            ("program", program),
            ("integration", integration),
            ("scene", scene),
            ("embodiment", embodiment),
            ("policy", execution_policy),
            ("deployment", deployment),
        )
    }
    task, policy = owned["integration"], owned["policy"]
    if "constraints" in task or "constraints.json" in task:
        raise ValueError(
            "Runtime constraints must be expressed through their existing owners."
        )
    profile = owned["embodiment"].get("skill_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("Bundle embodiment must declare its skill_profile.")
    selection = TaskProgramIntegrationCfg(
        robot_profile=profile["profile_id"],
        scene_registry=task["scene_binding"]["registry_id"],
        runtime_preset=policy["preset_id"],
    )
    compiled = compile_configured_task_program(
        owned["program"], scene_binding=task["scene_binding"], selection=selection
    )
    fingerprint = task_program_fingerprint(compiled)
    presets = dict(invocation_presets)
    if any(
        type(name) is not str
        or not name
        or name != name.strip()
        or type(value) is not str
        or not value
        or value != value.strip()
        for name, value in presets.items()
    ):
        raise ValueError(
            "Invocation presets require exact segment and preset identities."
        )
    bindings = []
    names = set()
    for segment in compiled.iter_segments():
        if (
            segment.name in names
            or segment.parallel_block is not None
            or segment.repeat_frames
            or len(segment.calls) != 1
        ):
            raise ValueError(
                "Generated bindings require uniquely named non-repeated serial single-call segments."
            )
        names.add(segment.name)
        if segment.name not in presets:
            raise ValueError(
                f"Missing explicit invocation preset for {segment.name!r}."
            )
        call = segment.calls[0]
        bindings.append(
            dict(
                program_fingerprint=fingerprint,
                invocation_id=f"{compiled.program_id}/{segment.segment_id}:{call.segment_call_index}",
                preset_id=presets[segment.name],
            )
        )
    if set(presets) != names:
        raise ValueError("Invocation presets contain unused segment names.")
    services = task.setdefault("runtime_services", {})
    if not isinstance(services, dict):
        raise TypeError("Runtime services must be a declaration mapping.")
    if "invocation_policy_bindings" in services:
        raise ValueError(
            "Invocation policies must have one declaration owner; supply segment presets instead."
        )
    services["invocation_policy_bindings"] = bindings
    configured = load_configured_task_program_deployment(
        task_program={
            "program": {"inline": owned["program"]},
            "integration": {"inline": task},
            "execution_policy": {"inline": policy},
        },
        skill_profile=profile,
        base_dir=base_dir,
    )
    final_program = configured.integration.registration.catalog.preflight(
        configured.load_program()
    )
    if task_program_fingerprint(final_program) != fingerprint:
        raise ValueError(
            "Effective program identity changed while binding declarations."
        )
    selected_graph["integration_fingerprint"] = (
        configured.integration.integration_fingerprint
    )
    selected_graph = validate_semantic_task_graph(selected_graph)
    files = {
        "task_program_deployment.yaml": yaml.safe_dump(
            owned["deployment"], sort_keys=False
        ).encode(),
        "task_program/program.yaml": yaml.safe_dump(
            owned["program"], sort_keys=False
        ).encode(),
        "task_program/integration.yaml": yaml.safe_dump(task, sort_keys=False).encode(),
        "components/scene.yaml": yaml.safe_dump(
            owned["scene"], sort_keys=False
        ).encode(),
        "components/embodiment.yaml": yaml.safe_dump(
            owned["embodiment"], sort_keys=False
        ).encode(),
        "components/execution_policy.yaml": yaml.safe_dump(
            policy, sort_keys=False
        ).encode(),
        "task_spec.json": _json(spec),
        "semantic_task_graph.json": _json(selected_graph),
    }
    for name, declaration in (artifact_declarations or {}).items():
        if (
            name in files
            or name in {"integration_fingerprint.json", SOURCE_INPUT_PROVENANCE_FILE}
            or not name.endswith(".json")
        ):
            raise ValueError(
                "Supplemental declarations cannot replace canonical bundle files."
            )
        files[name] = _json(declaration)
    snapshot = prepare_configured_bundle(
        files,
        base_dir=base_dir,
        external_input_digests=external_input_digests,
        implementation_revisions=implementation_revisions,
        source_input_provenance=source_input_provenance,
    )
    return selected_graph, snapshot


def publish_task_program_bundle(
    snapshot: BundleSnapshot,
    output_dir: str | Path,
    *,
    validate_inputs: Callable[[], None] | None = None,
) -> TaskProgramBundlePaths:
    """Atomically publish already preflighted declaration bytes to a fresh directory.

    Args:
        snapshot: Sealed source declarations returned by bundle preparation.
        output_dir: Fresh destination; existing paths are never overwritten.
        validate_inputs: Explicit source-owner check before writes and publication.

    Returns:
        Public canonical bundle paths. Existing videos/receipts/certificates are
        neither read nor copied; new execution owns independent run identities.
    """
    write_bundle_atomic(output_dir, snapshot, validate_inputs=validate_inputs)
    return TaskProgramBundlePaths.at(output_dir)
