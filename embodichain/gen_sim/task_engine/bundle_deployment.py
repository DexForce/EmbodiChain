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

"""Static preparation and loading of shared-runtime Task Program bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, TYPE_CHECKING

import yaml

from embodichain.gen_sim.task_spec import TaskSpec, canonical_hash
from embodichain.utils.config_paths import resolve_local_resource_paths
from .bundle_identity import _REQUIRED_FILES, create_bundle_identity
from .bundle_spec import bundle_task_spec_payload, decode_bundle_task_spec
from .bundle_inputs import (
    SOURCE_INPUT_PROVENANCE_FILE,
    SOURCE_INPUT_PROVENANCE_SCHEMA,
    _provenance_bytes,
)
from .bundle_io import BundleSnapshot, read_bundle_snapshot
from .program_declarations import _verify_program_projection

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
    from embodichain.lab.gym.utils.composition import ResolvedGymComponents
    from embodichain.lab.task_program import CompiledTaskProgram
    from embodichain.lab.task_program.integrations import (
        ConfiguredTaskProgramDeployment,
    )

__all__ = [
    "LoadedTaskProgramBundle",
    "prepare_configured_bundle",
    "load_configured_bundle",
]


class _DeclarationLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[str, Any]:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if type(key) is not str or key in result:
            raise ValueError("Bundle declarations require unique string mapping keys.")
        result[key] = loader.construct_object(value_node, deep=True)
    return result


_DeclarationLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _decode(content: bytes, name: str) -> dict[str, Any]:
    value = yaml.load(content.decode("utf-8"), Loader=_DeclarationLoader)
    if type(value) is not dict:
        raise ValueError(f"Bundle declaration {name!r} must be a mapping.")
    return value


@dataclass(frozen=True, slots=True)
class LoadedTaskProgramBundle:
    """Detached static declarations plus their sealed source bytes.

    This is not a simulator, run or physical certificate. Mutable physical and
    configured-deployment mappings are independently owned assembly results;
    changing effective declarations requires preparing a new bundle identity.
    """

    snapshot: BundleSnapshot
    physical: ResolvedGymComponents
    deployment: ConfiguredTaskProgramDeployment
    program: CompiledTaskProgram
    task_spec: TaskSpec | None
    root: Path
    _current_input_validator: Callable[[LoadedTaskProgramBundle], None] | None = field(
        default=None, repr=False, compare=False
    )

    def assert_current_inputs(self) -> None:
        """Recheck current-loader inputs and implementation before runtime use.

        The current-package loader binds this check to its independently hashed
        source files. Explicit qualification-owned loaders retain their caller's
        input contract. Sealed component files are never reopened; this reads
        only declared external inputs and the current implementation identity.
        """
        if self._current_input_validator is not None:
            self._current_input_validator(self)

    def configure_environment(self) -> tuple[str, EmbodiedEnvCfg]:
        """Build and register a Gym configuration from sealed bytes, without a simulator.

        Recompose from the snapshot instead of trusting mutable assembly copies
        exposed by this object. The returned configuration belongs to the caller;
        changing effective physical declarations requires a newly prepared bundle.
        """
        self.assert_current_inputs()
        from embodichain.lab.gym.utils.gym_utils import config_to_cfg

        snapshot = BundleSnapshot(self.snapshot.identity, self.snapshot.files)
        _, deployment, program, spec = _preflight(snapshot.files, self.root)
        _verify_identity(snapshot, deployment, program, spec)
        parsed = {
            name: _decode(content, name)
            for name, content in snapshot.files.items()
            if name.endswith((".yaml", ".yml"))
        }
        config = deepcopy(parsed["task_program_deployment.yaml"])
        config["task_program"] = {
            key: {"inline": parsed[path]}
            for key, path in config["task_program"].items()
        }
        cfg = config_to_cfg(
            config,
            source_path=self.root / "task_program_deployment.yaml",
            component_sources=parsed,
        )
        self.assert_current_inputs()
        return config["id"], cfg


def _validate_source_files(files: Mapping[str, bytes]) -> None:
    """Reject unsupported source layouts before resolving resource paths."""
    if "integration_fingerprint.json" in files:
        raise ValueError(
            "Prepare from source declarations, not an existing runtime envelope."
        )
    if {"constraints.json", "task_program/constraints.json"}.intersection(files):
        raise ValueError(
            "constraints.json is not supported; declare values through their owners."
        )
    if not _REQUIRED_FILES.issubset(files):
        raise ValueError("Bundle source declarations are incomplete.")
    if any(
        type(name) is not str or type(data) is not bytes for name, data in files.items()
    ):
        raise TypeError(
            "Bundle source declarations require string paths and exact bytes."
        )


def _preflight(
    files: Mapping[str, bytes],
    base_dir: Path,
) -> tuple[
    ResolvedGymComponents,
    ConfiguredTaskProgramDeployment,
    CompiledTaskProgram,
    TaskSpec | None,
]:
    # This validates source declarations, not an old bundle conversion format.
    _validate_source_files(files)
    parsed = {name: _decode(files[name], name) for name in _REQUIRED_FILES}
    declaration = parsed["task_program_deployment.yaml"]
    selectors = declaration.get("task_program")
    expected = {
        "program": "task_program/program.yaml",
        "integration": "task_program/integration.yaml",
        "execution_policy": "components/execution_policy.yaml",
    }
    if selectors != expected:
        raise ValueError(
            "Task Engine bundles must select their canonical materialized Task Program components."
        )
    from embodichain.lab.gym.utils.composition import resolve_gym_components
    from embodichain.lab.task_program.integrations import (
        load_configured_task_program_deployment,
    )
    from .semantic_graph import validate_semantic_task_graph

    sources = {
        name: _decode(data, name)
        for name, data in files.items()
        if name.endswith((".yaml", ".yml"))
    }
    physical = resolve_gym_components(
        declaration, base_dir=base_dir, component_sources=sources
    )
    batch_size = physical.config.get("num_envs", 1)
    if type(batch_size) is not int or batch_size != 1:
        raise ValueError(
            "Shared-runtime Task Engine bundles currently require num_envs=1."
        )
    if physical.embodiment_skill_profile is None:
        raise ValueError("Task Engine bundle requires an embodiment skill profile.")
    integration = parsed["task_program/integration.yaml"]
    for owner in (integration, physical.embodiment_skill_profile):
        services = owner.get("runtime_services", {})
        if isinstance(services, Mapping) and services.get("nominal_motion_sources"):
            raise ValueError(
                "Migrated Task Engine bundles do not support external nominal sources."
            )
    deployment = load_configured_task_program_deployment(
        task_program={key: {"inline": parsed[name]} for key, name in expected.items()},
        skill_profile=physical.embodiment_skill_profile,
        base_dir=base_dir,
    )
    physical.validate_scene_bindings(deployment.scene_binding)
    program = deployment.integration.registration.catalog.preflight(
        deployment.load_program(),
        invocation_policy_bindings=deployment.invocation_policy_bindings,
    )
    spec = decode_bundle_task_spec(parsed["task_spec.json"])
    graph = validate_semantic_task_graph(parsed["semantic_task_graph.json"])
    if (
        graph["integration_fingerprint"]
        != deployment.integration.integration_fingerprint
    ):
        raise ValueError(
            "Semantic graph has a different effective integration fingerprint."
        )
    _verify_program_projection(parsed["task_program/program.yaml"], graph)
    return physical, deployment, program, spec


def _identity_fields(
    deployment: ConfiguredTaskProgramDeployment,
    program: CompiledTaskProgram,
    spec: TaskSpec | None,
) -> dict[str, Any]:
    from embodichain.lab.task_program.integrations import task_program_fingerprint

    registration = deployment.integration.registration
    registration.assert_unchanged()
    return {
        "integration_id": deployment.integration_id,
        "integration_fingerprint": deployment.integration.integration_fingerprint,
        "registration_declaration": registration.fingerprint_payload(),
        "program_fingerprint": task_program_fingerprint(program),
        "task_spec_digest": canonical_hash(bundle_task_spec_payload(spec)),
    }


def _verify_identity(
    snapshot: BundleSnapshot,
    deployment: ConfiguredTaskProgramDeployment,
    program: CompiledTaskProgram,
    spec: TaskSpec | None,
) -> None:
    expected = snapshot.identity.to_dict()
    fields = _identity_fields(deployment, program, spec)
    fields["registration_fingerprint"] = canonical_hash(
        fields.pop("registration_declaration")
    )
    for name, value in fields.items():
        if expected[name] != value:
            raise ValueError(
                f"Effective bundle {name} differs from its sealed identity."
            )


def _materialize_physical_resources(files: dict[str, bytes], *, base_dir: Path) -> None:
    """Seal the public physical owner's selected, materialized declarations."""
    from embodichain.lab.gym.utils.composition import resolve_gym_components

    sources = {
        name: _decode(data, name)
        for name, data in files.items()
        if name.endswith((".yaml", ".yml"))
    }
    deployment_name = "task_program_deployment.yaml"
    deployment = resolve_local_resource_paths(
        sources[deployment_name], base_dir=base_dir
    )
    physical = resolve_gym_components(
        deployment, base_dir=base_dir, component_sources=sources
    )
    materialized = {deployment_name: deployment}
    for path, component in physical.materialized_component_sources.items():
        name = path.relative_to(base_dir).as_posix()
        materialized[name] = component
    for name, declaration in materialized.items():
        if declaration != sources[name]:
            files[name] = yaml.safe_dump(declaration, sort_keys=False).encode()


def prepare_configured_bundle(
    files: Mapping[str, bytes],
    *,
    base_dir: str | Path,
    external_input_digests: Mapping[str, str],
    implementation_revisions: Mapping[str, str],
    source_input_provenance: Mapping[str, Any] | None = None,
) -> BundleSnapshot:
    """Preflight pure owner declarations, then seal their identity without writing.

    Source files use the canonical Task Engine bundle layout. Explicit relative
    physical resources are resolved from each declaration's owner beneath
    base_dir before sealing, so publication does not change their location.
    Callers supply independently checked input digests. This entry validates
    graph projection and runtime capabilities, not physical success or TaskSpec
    certificate readiness.

    Args:
        files: Canonical source declaration bytes and optional audit artifacts.
        base_dir: Owner directory for resolving external resource declarations.
        external_input_digests: Independently acquired source-owner input hashes.
        implementation_revisions: Explicit current owner implementation revisions.
        source_input_provenance: Closed owner/path/SHA declaration. When omitted,
            application-owned inputs receive one explicit application group.

    Returns:
        Owned version-5 snapshot with exact external-input/provenance agreement.
    """
    owned = dict(files)
    if SOURCE_INPUT_PROVENANCE_FILE in owned and source_input_provenance is not None:
        raise ValueError("Source-input provenance already has a declaration owner.")
    provenance = (
        owned[SOURCE_INPUT_PROVENANCE_FILE]
        if SOURCE_INPUT_PROVENANCE_FILE in owned
        else (
            source_input_provenance
            if source_input_provenance is not None
            else {
                "schema_version": SOURCE_INPUT_PROVENANCE_SCHEMA,
                "groups": {"application": dict(external_input_digests)},
            }
        )
    )
    owned[SOURCE_INPUT_PROVENANCE_FILE] = _provenance_bytes(
        provenance, external_input_digests
    )
    _validate_source_files(owned)
    source_owner = Path(base_dir).expanduser().resolve()
    _materialize_physical_resources(owned, base_dir=source_owner)
    _, deployment, program, spec = _preflight(owned, source_owner)
    identity = create_bundle_identity(
        **_identity_fields(deployment, program, spec),
        implementation_revisions=implementation_revisions,
        component_files={
            name: hashlib.sha256(data).hexdigest() for name, data in owned.items()
        },
        external_inputs=external_input_digests,
    )
    return BundleSnapshot(identity, owned)


def load_configured_bundle(
    root: str | Path,
    *,
    external_input_digests: Mapping[str, str],
    implementation_revisions: Mapping[str, str],
) -> LoadedTaskProgramBundle:
    """Check the runtime envelope first, then recompose only its verified bytes.

    Reject old versions, changed inputs, unsupported capabilities and mismatched
    effective identities before creating any simulator or live adapter. Runtime
    registration factories remain lazy. Historical certificates are not loaded.
    """
    directory = Path(root).expanduser().resolve()
    snapshot = read_bundle_snapshot(
        directory,
        external_input_digests=external_input_digests,
        implementation_revisions=implementation_revisions,
    )
    physical, deployment, program, spec = _preflight(snapshot.files, directory)
    _verify_identity(snapshot, deployment, program, spec)
    return LoadedTaskProgramBundle(
        snapshot, physical, deployment, program, spec, directory
    )
