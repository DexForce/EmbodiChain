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

"""Static identities for the shared-runtime Task Engine bundle contract.

This module owns no runtime factories or filesystem access. The builder supplies
validated owner declarations and input digests; a loader checks this envelope
before importing or constructing a simulation deployment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath
from typing import Any, Final

from embodichain.gen_sim.task_spec import canonical_hash, canonical_json, json_snapshot
from .bundle_inputs import SOURCE_INPUT_PROVENANCE_FILE

__all__ = [
    "BUNDLE_IDENTITY_SCHEMA",
    "SHARED_RUNTIME_CONTRACT",
    "BundleIdentity",
    "create_bundle_identity",
    "verify_bundle_files",
]

BUNDLE_IDENTITY_SCHEMA: Final = "semantic_integration_fingerprint/v5"
SHARED_RUNTIME_CONTRACT: Final = "gen_sim.task_program/shared/v3"
_FIELDS = frozenset(
    {
        "schema_version",
        "adapter_contract",
        "integration_id",
        "integration_fingerprint",
        "registration_fingerprint",
        "program_fingerprint",
        "task_spec_digest",
        "implementation_revisions",
        "component_files",
        "external_inputs",
        "registration_declaration",
        "bundle_identity",
    }
)
_REQUIRED_FILES = frozenset(
    {
        "task_program_deployment.yaml",
        "task_program/program.yaml",
        "task_program/integration.yaml",
        "components/scene.yaml",
        "components/embodiment.yaml",
        "components/execution_policy.yaml",
        "task_spec.json",
        "semantic_task_graph.json",
        SOURCE_INPUT_PROVENANCE_FILE,
    }
)


def _identifier(value: object, name: str) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty exact string.")


def _digest(value: object, name: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")


def _validate(payload: dict[str, Any]) -> None:
    schema, contract = payload.get("schema_version"), payload.get("adapter_contract")
    if schema != BUNDLE_IDENTITY_SCHEMA or contract != SHARED_RUNTIME_CONTRACT:
        raise ValueError(
            f"Unsupported bundle runtime contract {contract!r} / {schema!r}. "
            "Rebuild from current task declarations with "
            "prepare_configured_bundle(); old bundles are not converted."
        )
    if set(payload) != _FIELDS:
        raise ValueError("Bundle identity has missing or unsupported fields.")
    _identifier(payload["integration_id"], "integration_id")
    for name in (
        "integration_fingerprint",
        "registration_fingerprint",
        "program_fingerprint",
        "task_spec_digest",
        "bundle_identity",
    ):
        _digest(payload[name], name)
    for name in ("component_files", "external_inputs", "implementation_revisions"):
        if type(payload[name]) is not dict:
            raise TypeError(f"{name} must be a JSON object.")
        for key, value in payload[name].items():
            _identifier(key, name)
            if name == "implementation_revisions":
                _identifier(value, f"{name}.{key}")
            else:
                _digest(value, f"{name}.{key}")
    if not payload["implementation_revisions"]:
        raise ValueError("Bundle identity requires implementation revisions.")
    files = payload["component_files"]
    if not _REQUIRED_FILES.issubset(files):
        raise ValueError("Bundle identity omits required declaration files.")
    for name in files:
        path = PurePosixPath(name)
        if (
            not path.parts
            or "\x00" in name
            or path.is_absolute()
            or str(path) != name
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or name == "integration_fingerprint.json"
        ):
            raise ValueError(
                f"Bundle file must be a normalized relative path: {name!r}."
            )
    declaration = payload["registration_declaration"]
    if type(declaration) is not dict or not declaration:
        raise ValueError("Complete registration declaration is required.")
    if canonical_hash(declaration) != payload["registration_fingerprint"]:
        raise ValueError("Registration declaration changed under its sealed identity.")
    unsigned = {
        key: value for key, value in payload.items() if key != "bundle_identity"
    }
    if canonical_hash(unsigned) != payload["bundle_identity"]:
        raise ValueError("Bundle declaration changed under its sealed identity.")


@dataclass(frozen=True, slots=True, init=False)
class BundleIdentity:
    """Owned canonical envelope, independent of simulation and mutable inputs.

    Use :meth:`from_dict` to verify an existing envelope, or
    :func:`create_bundle_identity` to seal a newly preflighted declaration.
    Identity changes do not authorize reuse of an earlier physical certificate.
    """

    _encoded: str

    def __init__(self) -> None:
        raise TypeError("Use BundleIdentity.from_dict() or create_bundle_identity().")

    @classmethod
    def from_dict(cls, value: BundleIdentity | Mapping[str, Any]) -> BundleIdentity:
        """Verify the version and hashes, rebuilding an independent snapshot."""
        payload = value.to_dict() if type(value) is cls else json_snapshot(value)
        if type(payload) is not dict:
            raise TypeError("Bundle identity must be a JSON object.")
        _validate(payload)
        result = object.__new__(cls)
        object.__setattr__(result, "_encoded", canonical_json(payload))
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return an independent serialized envelope without re-signing it."""
        return json.loads(self._encoded)

    @property
    def digest(self) -> str:
        """Return the sealed bundle identity digest."""
        return self.to_dict()["bundle_identity"]


def create_bundle_identity(
    *,
    integration_id: str,
    integration_fingerprint: str,
    registration_declaration: Mapping[str, Any],
    program_fingerprint: str,
    task_spec_digest: str,
    implementation_revisions: Mapping[str, str],
    component_files: Mapping[str, str],
    external_inputs: Mapping[str, str],
) -> BundleIdentity:
    """Seal a new owner-validated declaration without creating runtime objects.

    The caller must complete static deployment preflight first and enumerate
    every effective component and external input. Component values are byte
    digests of relative bundle files; external keys identify immutable inputs.
    This function validates identity integrity, not scene physics or capability
    completeness. Implementation revisions must describe the selected owners.
    """
    payload = json_snapshot(
        {
            "schema_version": BUNDLE_IDENTITY_SCHEMA,
            "adapter_contract": SHARED_RUNTIME_CONTRACT,
            "integration_id": integration_id,
            "integration_fingerprint": integration_fingerprint,
            "registration_declaration": registration_declaration,
            "registration_fingerprint": canonical_hash(registration_declaration),
            "program_fingerprint": program_fingerprint,
            "task_spec_digest": task_spec_digest,
            "implementation_revisions": implementation_revisions,
            "component_files": component_files,
            "external_inputs": external_inputs,
        }
    )
    payload["bundle_identity"] = canonical_hash(payload)
    return BundleIdentity.from_dict(payload)


def verify_bundle_files(
    identity: BundleIdentity | Mapping[str, Any], files: Mapping[str, bytes]
) -> None:
    """Require the exact sealed declaration file set and bytes before loading.

    The envelope itself and run outputs are not declaration files. The caller
    reads files without resolving undeclared links or external inputs; this
    function performs no I/O and never starts simulation.
    """
    expected = BundleIdentity.from_dict(identity).to_dict()["component_files"]
    if set(files) != set(expected):
        raise ValueError("Bundle declaration file set differs from its identity.")
    for name, content in files.items():
        if type(content) is not bytes:
            raise TypeError("Bundle declaration content must be exact bytes.")
        if hashlib.sha256(content).hexdigest() != expected[name]:
            raise ValueError(f"Bundle declaration file changed: {name}.")
