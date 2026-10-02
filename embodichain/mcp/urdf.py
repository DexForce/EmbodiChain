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

"""MCP adapter for simulation-independent URDF assembly workflows."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
import re
import shutil
import threading
import xml.etree.ElementTree as ET

import numpy as np

from .adapters import ToolRegistrar

__all__ = ["URDFAssemblyAdapter"]


AssetResolver = Callable[[str], str | Path]
SimulationVerifier = Callable[[Path, str, int | None], Mapping[str, object]]

_DEFAULT_ASSET_CATALOG: tuple[dict[str, str], ...] = (
    {
        "asset": "UniversalRobots/UR5/UR5.urdf",
        "component_type": "arm",
        "title": "Universal Robots UR5 arm",
    },
    {
        "asset": "DH_PGC_140_50_M/DH_PGC_140_50_M.urdf",
        "component_type": "hand",
        "title": "DH PGC 140-50-M gripper",
    },
    {
        "asset": "Franka/Panda/Panda.urdf",
        "component_type": "arm",
        "title": "Franka Panda arm",
    },
)
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_MAX_COMPONENTS = 16
_MAX_SENSORS = 32


def _resolve_data_asset(asset_ref: str) -> Path:
    """Resolve one relative registered data asset without accepting a path."""
    try:
        from embodichain.data import get_data_path

        resolved = Path(get_data_path(asset_ref)).expanduser().resolve()
    except (
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise FileNotFoundError(
            f"Registered URDF asset is unavailable: {asset_ref}"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(f"URDF asset does not resolve to a file: {asset_ref}")
    data_root = (
        Path(
            os.environ.get(
                "EMBODICHAIN_DATA_ROOT",
                str(Path.home() / ".cache" / "embodichain_data"),
            )
        )
        .expanduser()
        .resolve()
    )
    trusted_roots = (data_root, data_root / "extract")
    if not any(resolved == root or root in resolved.parents for root in trusted_roots):
        raise PermissionError(
            f"Resolved URDF asset escapes the configured data root: {asset_ref}"
        )
    return resolved


def _validate_asset_ref(value: object, *, field: str = "asset") -> str:
    """Validate a data-asset identifier before handing it to the resolver."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty asset identifier")
    if "\x00" in value:
        raise ValueError(f"{field} cannot contain NUL bytes")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            f"{field} must be a relative registered asset identifier, got {value!r}"
        )
    return value


def _normalize_transform(value: object, *, field: str) -> np.ndarray:
    """Normalize a JSON homogeneous transform and reject malformed matrices."""
    if value is None:
        return np.eye(4, dtype=float)
    try:
        transform = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a numeric 4x4 matrix") from error
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{field} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-6):
        raise ValueError(f"{field} must use homogeneous last row [0, 0, 0, 1]")
    return transform


def _safe_name(value: object, *, field: str) -> str:
    """Validate a user-facing name that will become part of an output path."""
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ValueError(
            f"{field} must match {_SAFE_NAME.pattern!r} and contain no path separators"
        )
    return value


def _json_digest(value: object) -> str:
    """Return a stable digest for one JSON-compatible assembly specification."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "URDF assembly specification must be JSON-compatible"
        ) from error
    return hashlib.sha256(encoded).hexdigest()[:16]


def _duplicates(values: Sequence[str]) -> list[str]:
    """Return duplicate names in first-seen order."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _sha256(path: Path) -> str:
    """Hash a generated artifact in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class URDFAssemblyAdapter:
    """Expose :class:`URDFAssemblyManager` through project-level MCP tools.

    The adapter owns only asset composition and generated-artifact handles.
    It does not create a simulator while composing or validating XML.  The
    optional ``urdf_verify_in_simulation`` tool invokes a separately scoped
    verifier after the pure assembly has completed.

    Args:
        output_root: Directory where generated URDFs and copied meshes are kept.
        asset_resolver: Resolver for relative data-asset identifiers.  The
            default uses :func:`embodichain.data.get_data_path` lazily.
        asset_catalog: Explicit allowlist of asset records.  The default uses
            the showcase assets and never accepts arbitrary cache paths.
        simulation_verifier: Optional callback used by tests or embedded hosts.
        max_assemblies: Maximum number of active manifests and generated
            assembly directories retained by this adapter.
    """

    name = "urdf"

    def __init__(
        self,
        *,
        output_root: str | Path | None = None,
        asset_resolver: AssetResolver | None = None,
        asset_catalog: Sequence[Mapping[str, str]] | None = None,
        simulation_verifier: SimulationVerifier | None = None,
        max_assemblies: int = 64,
    ) -> None:
        if type(max_assemblies) is not int or max_assemblies < 1:
            raise ValueError("max_assemblies must be a positive integer")
        self.output_root = (
            (Path(output_root) if output_root is not None else Path("outputs/mcp/urdf"))
            .expanduser()
            .resolve()
        )
        self._asset_resolver = asset_resolver or _resolve_data_asset
        catalog = _DEFAULT_ASSET_CATALOG if asset_catalog is None else asset_catalog
        self._asset_catalog = self._normalize_asset_catalog(catalog)
        self._allowed_assets = frozenset(item["asset"] for item in self._asset_catalog)
        self._simulation_verifier = simulation_verifier
        self._max_assemblies = max_assemblies
        self._assemblies: dict[str, dict[str, object]] = {}
        self._lock = threading.RLock()
        self._prune_output_root()

    def close(self) -> None:
        """Release in-memory handles while preserving generated artifacts."""
        with self._lock:
            self._assemblies.clear()

    def capabilities(self) -> Sequence[str]:
        """Return operation names contributed by the URDF domain."""
        return (
            "urdf_list_assets",
            "urdf_compose",
            "urdf_inspect",
            "urdf_validate",
            "urdf_verify_in_simulation",
        )

    def register(self, server: object, *, register_tool: ToolRegistrar) -> None:
        """Register URDF tools, resources, and the composition prompt."""
        register_tool(
            "urdf_list_assets",
            self.list_assets,
            read_only=True,
            idempotent=True,
        )
        register_tool(
            "urdf_compose",
            self.compose,
            read_only=False,
            destructive=True,
            idempotent=False,
        )
        register_tool(
            "urdf_inspect",
            self.inspect,
            read_only=True,
            idempotent=True,
        )
        register_tool(
            "urdf_validate",
            self.validate,
            read_only=True,
            idempotent=True,
        )
        register_tool(
            "urdf_verify_in_simulation",
            self.verify_in_simulation,
            read_only=False,
            destructive=True,
            idempotent=False,
        )

        @server.resource(
            "embodichain://urdf/assets/catalog", mime_type="application/json"
        )
        def urdf_asset_catalog() -> str:
            return json.dumps(self.list_assets(), ensure_ascii=False, indent=2)

        @server.resource(
            "embodichain://urdf/assemblies/{assembly_id}/manifest",
            mime_type="application/json",
        )
        def urdf_manifest(assembly_id: str) -> str:
            return json.dumps(self._manifest(assembly_id), ensure_ascii=False, indent=2)

        @server.resource(
            "embodichain://urdf/assemblies/{assembly_id}/model",
            mime_type="application/xml",
        )
        def urdf_model(assembly_id: str) -> str:
            return self._model(assembly_id)

        @server.resource(
            "embodichain://urdf/assemblies/{assembly_id}/validation",
            mime_type="application/json",
        )
        def urdf_validation(assembly_id: str) -> str:
            return json.dumps(self.validate(assembly_id), ensure_ascii=False, indent=2)

        @server.prompt(name="compose_urdf")
        def compose_urdf_prompt() -> str:
            return (
                "Compose a modular robot from registered URDF asset identifiers. "
                "Start with urdf_list_assets, call urdf_compose with component_type "
                "and asset fields, read the manifest and model resources, then run "
                "urdf_validate. Call urdf_verify_in_simulation only when the user "
                "requests backend verification."
            )

    def list_assets(self) -> dict[str, object]:
        """List the built-in URDF asset references used by the showcase.

        Returns:
            A JSON-compatible catalog.  Availability is a local-cache probe and
            does not download an asset.
        """
        assets = []
        for item in self._asset_catalog:
            record = dict(item)
            record["available"] = self._is_locally_available(item["asset"])
            assets.append(record)
        return {"assets": assets, "scope": "registered_data_assets"}

    def compose(
        self,
        components: Sequence[Mapping[str, object]],
        *,
        sensors: Sequence[Mapping[str, object]] | None = None,
        assembly_name: str = "assembly",
        name_case: Mapping[str, str] | None = None,
        component_prefix: Sequence[Sequence[str | None]] | None = None,
        use_signature_check: bool = True,
    ) -> dict[str, object]:
        """Compose registered URDF components into one generated model.

        Args:
            components: Component records with ``component_type`` and ``asset``.
                An optional finite 4x4 ``transform`` and JSON ``params`` mapping
                may be supplied.
            sensors: Optional sensor records with ``sensor_name``, ``asset``,
                ``parent_component``, and ``parent_link``.
            assembly_name: Safe output basename without a path component.
            name_case: Optional link/joint casing policy forwarded to the toolkit.
            component_prefix: Optional ``(component_type, prefix)`` pairs.
            use_signature_check: Reuse an unchanged generated assembly when possible.

        Returns:
            An assembly handle, manifest URI, model URI, and validation report.

        Raises:
            ValueError: If the specification is malformed or an asset cannot be
                resolved.
        """
        normalized_components = self._normalize_components(components)
        normalized_sensors = self._normalize_sensors(sensors)
        name = _safe_name(assembly_name, field="assembly_name")
        normalized_prefix = self._normalize_prefix(component_prefix)
        normalized_case = self._normalize_name_case(name_case)
        specification = {
            "components": normalized_components,
            "sensors": normalized_sensors,
            "assembly_name": name,
            "name_case": normalized_case,
            "component_prefix": normalized_prefix,
        }
        assembly_id = f"assembly-{_json_digest(specification)}"
        output_dir = self.output_root / assembly_id
        output_path = output_dir / f"{name}.urdf"

        from embodichain.toolkits.urdf_assembly import URDFAssemblyManager

        manager = URDFAssemblyManager()
        if normalized_case is not None:
            manager.name_case = normalized_case
        if normalized_prefix is not None:
            manager.component_prefix = [tuple(item) for item in normalized_prefix]

        resolved_components: list[dict[str, object]] = []
        for component in normalized_components:
            resolved_path = self._resolve_asset(component["asset"])
            params = dict(component["params"])
            if not manager.add_component(
                str(component["component_type"]),
                str(resolved_path),
                np.asarray(component["transform"], dtype=float),
                **params,
            ):
                raise ValueError(
                    f"Unable to register URDF component {component['component_type']!r}"
                )
            resolved_components.append(deepcopy(component))

        resolved_sensors: list[dict[str, object]] = []
        for sensor in normalized_sensors:
            resolved_path = self._resolve_asset(sensor["asset"])
            params = dict(sensor["params"])
            if sensor["sensor_type"] is not None:
                params["sensor_type"] = sensor["sensor_type"]
            if not manager.attach_sensor(
                sensor_name=str(sensor["sensor_name"]),
                sensor_source=str(resolved_path),
                parent_component=str(sensor["parent_component"]),
                parent_link=str(sensor["parent_link"]),
                transform=np.asarray(sensor["transform"], dtype=float),
                **params,
            ):
                raise ValueError(f"Unable to attach sensor {sensor['sensor_name']!r}")
            resolved_sensors.append(deepcopy(sensor))

        output_dir_existed = output_dir.exists()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            manager.merge_urdfs(
                output_path=str(output_path),
                use_signature_check=use_signature_check,
            )
            manifest = self._make_manifest(
                assembly_id,
                output_path,
                resolved_components,
                resolved_sensors,
                specification,
            )
        except Exception:
            if not output_dir_existed:
                shutil.rmtree(output_dir, ignore_errors=True)
            raise
        with self._lock:
            if assembly_id not in self._assemblies:
                while len(self._assemblies) >= self._max_assemblies:
                    evicted_id = next(iter(self._assemblies))
                    evicted = self._assemblies.pop(evicted_id)
                    self._remove_artifacts(evicted)
            self._assemblies[assembly_id] = manifest
        return deepcopy(manifest)

    def inspect(self, assembly_id: str) -> dict[str, object]:
        """Return the stored manifest for one assembly handle."""
        return deepcopy(self._manifest(assembly_id))

    def validate(self, assembly_id: str) -> dict[str, object]:
        """Run the pure XML, topology, and mesh-reference checks again."""
        with self._lock:
            record = self._get_record(assembly_id)
            output_path = Path(str(record["output_path"]))
            validation = self._validate_model(output_path)
            record["validation"] = validation
            return deepcopy(validation | {"assembly_id": assembly_id})

    def verify_in_simulation(
        self,
        assembly_id: str,
        *,
        backend: str = "default",
        seed: int | None = None,
    ) -> dict[str, object]:
        """Optionally load a generated URDF into a temporary simulation world.

        The generated file is never reassembled by this operation.  A verifier
        can be injected for protocol tests; otherwise the simulation backend is
        imported lazily and the temporary world is destroyed before returning.
        """
        if backend not in {"default", "newton"}:
            raise ValueError("backend must be 'default' or 'newton'")
        with self._lock:
            record = self._get_record(assembly_id)
            output_path = Path(str(record["output_path"]))
            validation = self._validate_model(output_path)
            record["validation"] = validation
        if not bool(validation.get("valid")):
            raise ValueError(
                f"Cannot verify invalid URDF assembly {assembly_id!r}; validate it first"
            )

        verifier = self._simulation_verifier
        if verifier is None:
            result = {
                "status": "failed",
                "backend": backend,
                "error": "No simulation verifier is configured for this adapter",
            }
            result["assembly_id"] = assembly_id
            with self._lock:
                self._get_record(assembly_id)["simulation_verification"] = deepcopy(
                    result
                )
            return result
        try:
            result = dict(verifier(output_path, backend, seed))
        except Exception as error:  # noqa: BLE001 - report optional verifier failure.
            result = {
                "status": "failed",
                "backend": backend,
                "error": f"{type(error).__name__}: {error}",
            }
        result.setdefault("status", "succeeded")
        result["assembly_id"] = assembly_id
        with self._lock:
            self._get_record(assembly_id)["simulation_verification"] = deepcopy(result)
        return result

    def _normalize_components(
        self, components: Sequence[Mapping[str, object]]
    ) -> list[dict[str, object]]:
        """Normalize component records into the adapter's JSON contract."""
        if isinstance(components, (str, bytes)) or not isinstance(components, Sequence):
            raise ValueError("components must be a non-empty sequence of mappings")
        if not components or len(components) > _MAX_COMPONENTS:
            raise ValueError(f"components must contain 1 to {_MAX_COMPONENTS} entries")
        normalized: list[dict[str, object]] = []
        component_types: set[str] = set()
        for index, value in enumerate(components):
            if not isinstance(value, Mapping):
                raise ValueError(f"components[{index}] must be a mapping")
            component_type = value.get("component_type", value.get("type"))
            if not isinstance(component_type, str) or not component_type:
                raise ValueError(
                    f"components[{index}].component_type must be a non-empty string"
                )
            if component_type in component_types:
                raise ValueError(
                    f"components cannot contain duplicate component_type {component_type!r}"
                )
            component_types.add(component_type)
            asset = _validate_asset_ref(value.get("asset", value.get("asset_id")))
            params = value.get("params", {})
            if not isinstance(params, Mapping):
                raise ValueError(f"components[{index}].params must be a mapping")
            normalized.append(
                {
                    "component_type": component_type,
                    "asset": asset,
                    "transform": _normalize_transform(
                        value.get("transform"),
                        field=f"components[{index}].transform",
                    ).tolist(),
                    "params": deepcopy(dict(params)),
                }
            )
        return normalized

    def _normalize_sensors(
        self, sensors: Sequence[Mapping[str, object]] | None
    ) -> list[dict[str, object]]:
        """Normalize optional sensor attachment records."""
        if sensors is None:
            return []
        if isinstance(sensors, (str, bytes)) or not isinstance(sensors, Sequence):
            raise ValueError("sensors must be a sequence of mappings")
        if len(sensors) > _MAX_SENSORS:
            raise ValueError(f"sensors cannot contain more than {_MAX_SENSORS} entries")
        normalized: list[dict[str, object]] = []
        sensor_names: set[str] = set()
        for index, value in enumerate(sensors):
            if not isinstance(value, Mapping):
                raise ValueError(f"sensors[{index}] must be a mapping")
            required = ("sensor_name", "parent_component", "parent_link")
            missing = [key for key in required if not isinstance(value.get(key), str)]
            if missing:
                raise ValueError(
                    f"sensors[{index}] is missing string fields: {', '.join(missing)}"
                )
            sensor_name = str(value["sensor_name"])
            if sensor_name in sensor_names:
                raise ValueError(
                    f"sensors cannot contain duplicate sensor_name {sensor_name!r}"
                )
            sensor_names.add(sensor_name)
            params = value.get("params", {})
            if not isinstance(params, Mapping):
                raise ValueError(f"sensors[{index}].params must be a mapping")
            normalized.append(
                {
                    "sensor_name": sensor_name,
                    "asset": _validate_asset_ref(
                        value.get("asset", value.get("asset_id")),
                        field=f"sensors[{index}].asset",
                    ),
                    "parent_component": str(value["parent_component"]),
                    "parent_link": str(value["parent_link"]),
                    "sensor_type": value.get("sensor_type"),
                    "transform": _normalize_transform(
                        value.get("transform"),
                        field=f"sensors[{index}].transform",
                    ).tolist(),
                    "params": deepcopy(dict(params)),
                }
            )
        return normalized

    @staticmethod
    def _normalize_name_case(
        value: Mapping[str, str] | None,
    ) -> dict[str, str] | None:
        """Validate an optional link/joint case mapping."""
        if value is None:
            return None
        result = dict(value)
        if set(result) != {"joint", "link"} or any(
            mode not in {"upper", "lower", "original", "none"}
            for mode in result.values()
        ):
            raise ValueError(
                "name_case must contain only joint/link keys with upper, lower, "
                "original, or none values"
            )
        return result

    @staticmethod
    def _normalize_prefix(
        value: Sequence[Sequence[str | None]] | None,
    ) -> list[list[str | None]] | None:
        """Validate JSON ``component_prefix`` pairs."""
        if value is None:
            return None
        result: list[list[str | None]] = []
        component_names: set[str] = set()
        for index, item in enumerate(value):
            if isinstance(item, (str, bytes)) or len(item) != 2:
                raise ValueError(f"component_prefix[{index}] must contain two values")
            component, prefix = item
            if not isinstance(component, str) or not (
                prefix is None or isinstance(prefix, str)
            ):
                raise ValueError(
                    f"component_prefix[{index}] must be (string, string | null)"
                )
            if component in component_names:
                raise ValueError(
                    f"component_prefix cannot repeat component {component!r}"
                )
            component_names.add(component)
            result.append([component, prefix])
        return result

    def _resolve_asset(self, asset_ref: object) -> Path:
        """Resolve and normalize a validated data-asset identifier."""
        value = _validate_asset_ref(asset_ref)
        if value not in self._allowed_assets:
            raise ValueError(f"URDF asset is not registered: {value}")
        try:
            path = Path(self._asset_resolver(value)).expanduser().resolve()
        except (
            AttributeError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            raise FileNotFoundError(
                f"Registered URDF asset is unavailable: {value}"
            ) from error
        if not path.is_file():
            raise FileNotFoundError(f"Resolved URDF asset is not a file: {value}")
        return path

    @staticmethod
    def _normalize_asset_catalog(
        catalog: Sequence[Mapping[str, str]],
    ) -> tuple[dict[str, str], ...]:
        """Normalize and validate the MCP asset allowlist."""
        if isinstance(catalog, (str, bytes)) or not isinstance(catalog, Sequence):
            raise ValueError("asset_catalog must be a non-empty sequence of mappings")
        if not catalog:
            raise ValueError("asset_catalog must contain at least one asset")
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, item in enumerate(catalog):
            if not isinstance(item, Mapping):
                raise ValueError(f"asset_catalog[{index}] must be a mapping")
            asset = _validate_asset_ref(
                item.get("asset"), field=f"asset_catalog[{index}].asset"
            )
            if asset in seen:
                raise ValueError(f"asset_catalog contains duplicate asset {asset!r}")
            component_type = item.get("component_type", "")
            title = item.get("title", asset)
            if not isinstance(component_type, str) or not component_type:
                raise ValueError(
                    f"asset_catalog[{index}].component_type must be a non-empty string"
                )
            if not isinstance(title, str) or not title:
                raise ValueError(
                    f"asset_catalog[{index}].title must be a non-empty string"
                )
            seen.add(asset)
            normalized.append(
                {"asset": asset, "component_type": component_type, "title": title}
            )
        return tuple(normalized)

    def _prune_output_root(self) -> None:
        """Keep only the newest bounded set of generated assembly directories."""
        if not self.output_root.is_dir():
            return
        directories = sorted(
            (
                path
                for path in self.output_root.iterdir()
                if path.is_dir() and path.name.startswith("assembly-")
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in directories[self._max_assemblies :]:
            shutil.rmtree(path, ignore_errors=True)

    @staticmethod
    def _is_locally_available(asset_ref: str) -> bool:
        """Probe common data-cache locations without downloading anything."""
        root = Path(
            os.environ.get(
                "EMBODICHAIN_DATA_ROOT",
                str(Path.home() / ".cache" / "embodichain_data"),
            )
        )
        return any((base / asset_ref).is_file() for base in (root, root / "extract"))

    def _make_manifest(
        self,
        assembly_id: str,
        output_path: Path,
        components: list[dict[str, object]],
        sensors: list[dict[str, object]],
        specification: dict[str, object],
    ) -> dict[str, object]:
        """Build the JSON-safe handle manifest for a generated model."""
        validation = self._validate_model(output_path)
        return {
            "status": "succeeded",
            "assembly_id": assembly_id,
            "assembly_name": specification["assembly_name"],
            "components": deepcopy(components),
            "sensors": deepcopy(sensors),
            "model": {
                "path": str(output_path),
                "sha256": _sha256(output_path),
                "resource_uri": f"embodichain://urdf/assemblies/{assembly_id}/model",
            },
            "manifest_uri": (f"embodichain://urdf/assemblies/{assembly_id}/manifest"),
            "validation_uri": (
                f"embodichain://urdf/assemblies/{assembly_id}/validation"
            ),
            "specification": deepcopy(specification),
            "validation": validation,
            "simulation_verification": None,
            "output_path": str(output_path),
        }

    def _validate_model(self, output_path: Path) -> dict[str, object]:
        """Validate XML structure, topology, mimic references, and mesh files."""
        diagnostics: list[dict[str, str]] = []
        if not output_path.is_file():
            return {
                "valid": False,
                "errors": [{"code": "missing_model", "message": str(output_path)}],
                "warnings": [],
                "link_count": 0,
                "joint_count": 0,
                "root_links": [],
                "mesh_files": [],
                "missing_mesh_files": [],
            }
        try:
            root = ET.parse(output_path).getroot()
        except (ET.ParseError, OSError) as error:
            return {
                "valid": False,
                "errors": [
                    {
                        "code": "invalid_xml",
                        "message": f"{type(error).__name__}: {error}",
                    }
                ],
                "warnings": [],
                "link_count": 0,
                "joint_count": 0,
                "root_links": [],
                "mesh_files": [],
                "missing_mesh_files": [],
            }

        if root.tag != "robot":
            diagnostics.append(
                {
                    "severity": "error",
                    "code": "invalid_root",
                    "message": "root tag must be robot",
                }
            )
        links = root.findall("link")
        joints = root.findall("joint")
        link_names = [link.get("name", "") for link in links]
        joint_names = [joint.get("name", "") for joint in joints]
        errors = [item for item in diagnostics if item["severity"] == "error"]
        warnings: list[dict[str, str]] = []
        for kind, names in (("link", link_names), ("joint", joint_names)):
            if any(not name for name in names):
                errors.append(
                    {
                        "code": f"unnamed_{kind}",
                        "message": f"every {kind} must have a name",
                    }
                )
            for duplicate in _duplicates(names):
                errors.append(
                    {
                        "code": f"duplicate_{kind}",
                        "message": f"duplicate {kind} name: {duplicate}",
                    }
                )

        link_set = {name for name in link_names if name}
        child_links: set[str] = set()
        for joint in joints:
            parent = joint.find("parent")
            child = joint.find("child")
            parent_name = None if parent is None else parent.get("link")
            child_name = None if child is None else child.get("link")
            if parent_name not in link_set or child_name not in link_set:
                errors.append(
                    {
                        "code": "missing_joint_link",
                        "message": f"joint {joint.get('name', '<unnamed>')} references a missing parent or child link",
                    }
                )
            if child_name:
                child_links.add(child_name)
            mimic = joint.find("mimic")
            if mimic is not None and mimic.get("joint") not in set(joint_names):
                errors.append(
                    {
                        "code": "missing_mimic_joint",
                        "message": f"joint {joint.get('name', '<unnamed>')} references missing mimic joint {mimic.get('joint')!r}",
                    }
                )
        root_links = sorted(link_set - child_links)
        if len(root_links) != 1:
            errors.append(
                {
                    "code": "invalid_root_count",
                    "message": f"expected one root link, found {len(root_links)}",
                }
            )

        mesh_files = sorted(
            {
                str(mesh.get("filename"))
                for mesh in root.findall(".//mesh")
                if mesh.get("filename")
            }
        )
        missing_mesh_files: list[str] = []
        for mesh_file in mesh_files:
            if mesh_file.startswith("package://"):
                relative = mesh_file.removeprefix("package://")
            else:
                relative = mesh_file
            mesh_path = Path(relative)
            if mesh_path.is_absolute() or ".." in mesh_path.parts:
                errors.append(
                    {
                        "code": "unsafe_mesh_path",
                        "message": f"mesh reference escapes the generated assembly: {mesh_file}",
                    }
                )
                missing_mesh_files.append(mesh_file)
                continue
            mesh_path = (output_path.parent / mesh_path).resolve()
            if output_path.parent.resolve() not in mesh_path.parents:
                errors.append(
                    {
                        "code": "unsafe_mesh_path",
                        "message": f"mesh reference escapes the generated assembly: {mesh_file}",
                    }
                )
                missing_mesh_files.append(mesh_file)
                continue
            if not mesh_path.is_file():
                missing_mesh_files.append(mesh_file)
        if missing_mesh_files:
            errors.append(
                {
                    "code": "missing_mesh_files",
                    "message": f"missing {len(missing_mesh_files)} referenced mesh file(s)",
                }
            )

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "link_count": len(links),
            "joint_count": len(joints),
            "root_links": root_links,
            "mesh_files": mesh_files,
            "missing_mesh_files": missing_mesh_files,
        }

    def _get_record(self, assembly_id: str) -> dict[str, object]:
        """Return one in-memory assembly record or raise a protocol error."""
        try:
            return self._assemblies[assembly_id]
        except KeyError as error:
            raise ValueError(f"Unknown assembly_id: {assembly_id}") from error

    def _remove_artifacts(self, record: Mapping[str, object]) -> None:
        """Remove one bounded assembly directory when its handle is evicted."""
        output_path = Path(str(record.get("output_path", ""))).resolve()
        output_dir = output_path.parent
        if output_dir.parent != self.output_root or not output_dir.is_dir():
            return
        shutil.rmtree(output_dir, ignore_errors=True)

    def _manifest(self, assembly_id: str) -> dict[str, object]:
        """Return a manifest with the latest validation result."""
        with self._lock:
            record = self._get_record(assembly_id)
            record["validation"] = self._validate_model(
                Path(str(record["output_path"]))
            )
            return deepcopy(record)

    def _model(self, assembly_id: str) -> str:
        """Return generated URDF XML for a resource request."""
        with self._lock:
            record = self._get_record(assembly_id)
            path = Path(str(record["output_path"]))
        try:
            return path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(f"Unable to read generated URDF: {path}") from error
