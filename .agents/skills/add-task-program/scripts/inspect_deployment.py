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

"""Inspect configured Task Program deployments without constructing a simulator."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.lab.gym.utils.composition import ResolvedGymComponents
    from embodichain.lab.task_program.integrations.composition import (
        ConfiguredTaskProgramDeployment,
    )

__all__ = ["inspect_deployment", "main"]


def _has_path(value: object, path: tuple[str, ...]) -> bool:
    """Check an authored field without guessing through scalar overrides."""
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return False
        value = value[key]
    return True


def _configuration_view(
    payload: Mapping[str, object],
    physical: ResolvedGymComponents,
    deployment: ConfiguredTaskProgramDeployment,
    path: Path,
) -> dict[str, object]:
    """Explain production-composed declarations and their authored owners."""
    from dataclasses import asdict, fields, is_dataclass

    from embodichain.lab.task_program.integrations._configured_composition import (
        _merge_runtime_services,
        _resolve_task_program_components,
    )
    from embodichain.lab.task_program.integrations.catalog import _canonical_value
    from embodichain.lab.task_program.integrations.configured import (
        _normalize_skill_options_key,
    )
    from embodichain.lab.task_program.semantics.effects import CompositeEffectMonitorCfg
    from embodichain.utils.config_paths import resolve_config_path
    from embodichain.utils.utility import load_config

    task_program = physical.config["task_program"]
    _, integration, policy = _resolve_task_program_components(
        task_program, base_dir=path.parent
    )

    def source(value: object, prefix: tuple[str, ...]) -> tuple[Path, tuple[str, ...]]:
        if isinstance(value, Mapping):
            if "component" in value:
                value = value["component"]
            else:
                return path, prefix + (("inline",) if "inline" in value else ())
        if isinstance(value, str):
            return resolve_config_path(value, base_dir=path.parent).resolve(), ()
        return path, prefix

    environment = payload.get("environment")
    if isinstance(environment, Mapping) and "component" not in environment:
        variants = environment.get("variants", environment)
        environment = variants[physical.config["physics"]]
    environment_source, _ = source(environment, ())
    embodiment_source, _ = source(payload.get("embodiment"), ("robot",))
    integration_selector = task_program["integration"]
    policy_selector = task_program["execution_policy"]
    integration_source, integration_prefix = source(
        integration_selector, ("task_program", "integration")
    )
    policy_source, policy_prefix = source(
        policy_selector, ("task_program", "execution_policy")
    )
    authored_integration = (
        load_config(integration_source)
        if integration_source != path
        else integration_selector.get("inline", integration_selector)
    )
    authored_policy = (
        load_config(policy_source)
        if policy_source != path
        else policy_selector.get("inline", policy_selector)
    )
    integration = dict(integration)
    integration["profile"] = _normalize_skill_options_key(
        integration["profile"], path="integration.profile"
    )
    parameters: list[dict[str, object]] = []

    def origin(scope: str, keys: tuple[str, ...]) -> dict[str, object]:
        selector = integration_selector if scope == "integration" else policy_selector
        patch = selector.get("overrides", {}) if isinstance(selector, Mapping) else {}
        if _has_path(patch, keys):
            return {
                "file": str(path),
                "path": ["task_program", scope, "overrides", *keys],
                "kind": "deployment_override",
            }
        owner, prefix = (
            (integration_source, integration_prefix)
            if scope == "integration"
            else (policy_source, policy_prefix)
        )
        authored_keys = keys
        if (
            scope == "integration"
            and keys[:2] == ("profile", "skill_options")
            and "action_options" in authored_integration.get("profile", {})
        ):
            authored_keys = ("profile", "action_options", *keys[2:])
        declaration = (
            authored_integration if scope == "integration" else authored_policy
        )
        kind = "component" if owner != path else "inline"
        if not _has_path(declaration, authored_keys):
            kind = "derived"
            while authored_keys and not _has_path(declaration, authored_keys):
                authored_keys = authored_keys[:-1]
        return {
            "file": str(owner),
            "path": [*prefix, *authored_keys],
            "kind": kind,
        }

    def collect(scope: str, value: object, keys: tuple[str, ...] = ()) -> None:
        if isinstance(value, Mapping) and value:
            for key, child in value.items():
                collect(scope, child, (*keys, key))
        else:
            parameters.append(
                {
                    "scope": scope,
                    "path": list(keys),
                    "value": _canonical_value(value),
                    "source": origin(scope, keys),
                }
            )

    for key in ("requires", "profile"):
        collect("integration", integration[key], (key,))
    collect("execution_policy", policy)
    monitor_parameters: list[dict[str, object]] = []
    presets = deployment.integration.registration.robot_profile_binding.presets
    for name, ref in presets[0].effect_monitors.items():
        if ref.monitor_id != "builtin.composite_effect" or ref.revision != "1":
            continue
        configured = ref.params
        resolved = asdict(CompositeEffectMonitorCfg(**configured))
        for field, value in resolved.items():
            keys = ("profile", "effect_monitors", name, "params", field)
            monitor_parameters.append(
                {
                    "monitor": name,
                    "parameter": field,
                    "value": value,
                    "source": (
                        origin("integration", keys)
                        if field in configured
                        else {
                            "kind": "sdk_default",
                            "type": "CompositeEffectMonitorCfg",
                            "field": field,
                        }
                    ),
                }
            )

    def unpack(value: object) -> object:
        # Fingerprint serialization intentionally omits inactive extensions;
        # inspection must retain those SDK defaults as well.
        if is_dataclass(value) and not isinstance(value, type):
            return {
                field.name: unpack(getattr(value, field.name))
                for field in fields(value)
            }
        if isinstance(value, Mapping):
            return {key: unpack(child) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            return [unpack(child) for child in value]
        return value

    service_sources = {}
    if physical.embodiment_skill_profile.get("runtime_services"):
        service_sources["embodiment"] = {
            "file": str(embodiment_source),
            "path": ["skill_profile", "runtime_services"],
        }
    if integration.get("runtime_services"):
        service_sources["integration"] = {
            "file": str(integration_source),
            "path": [*integration_prefix, "runtime_services"],
        }

    return {
        "components": {
            "deployment": str(path),
            "environment": str(environment_source),
            "embodiment": str(embodiment_source),
            "integration": str(integration_source),
            "execution_policy": str(policy_source),
            "program": (
                str(deployment.program_path) if deployment.program_path else str(path)
            ),
        },
        "physics": {
            "value": physical.config.get("physics"),
            "source": {"file": str(environment_source), "path": ["physics"]},
        },
        "parameters": parameters,
        "resources": physical.embodiment_skill_profile["resources"],
        "resource_source": {
            "file": str(embodiment_source),
            "path": ["skill_profile", "resources"],
        },
        "runtime_services": {
            "values": _canonical_value(
                _merge_runtime_services(
                    physical.embodiment_skill_profile.get("runtime_services", {}),
                    integration.get("runtime_services", {}),
                )
            ),
            "sources": service_sources,
        },
        "runtime_presets": _canonical_value(unpack(presets)),
        "monitor_parameters": monitor_parameters,
        "scope": (
            "Parameters trace composed integration/policy declarations. Runtime presets "
            "also include SDK option defaults. Built-in composite monitor defaults "
            "are resolved without physical evidence; other monitor defaults require "
            "their provider. No simulator or runtime measurement is constructed."
        ),
    }


def inspect_deployment(
    path: str | Path, *, explain: bool = False, strict_contracts: bool = False
) -> dict[str, object]:
    """Decode, compose, and preflight one configured Task Program deployment.

    Args:
        path: Runnable Gym deployment containing a `task_program` selection.
        explain: Include normalized declarations, resolved values and authored owners.
        strict_contracts: Require an instruction and acceptance contract on each segment.

    Returns:
        JSON-compatible program statistics and optional composed values/owners.

    Raises:
        FileNotFoundError: If the deployment or one selected component is absent.
        TypeError: If a decoded configuration value has the wrong shape.
        ValueError: If component, contract, scene, or program validation fails.
    """
    deployment_path = Path(path).expanduser().resolve()
    if not deployment_path.is_file():
        raise FileNotFoundError(
            f"Task Program deployment is not a file: {deployment_path}."
        )
    from embodichain.lab.gym.utils._component_composition import (
        _resolve_gym_components,
        _validate_scene_binding_targets,
    )
    from embodichain.lab.task_program.integrations._configured_composition import (
        _load_configured_task_program_deployment,
    )
    from embodichain.utils.utility import load_config

    payload = load_config(deployment_path)
    if not isinstance(payload, Mapping):
        raise TypeError("Runnable Task Program deployment must decode to a mapping.")

    physical = _resolve_gym_components(
        payload,
        base_dir=deployment_path.parent,
    )
    task_program = physical.config.get("task_program")
    if task_program is None:
        raise ValueError("Runnable deployment does not declare task_program.")
    if physical.embodiment_skill_profile is None:
        raise ValueError(
            "Configured Task Program deployment must select an embodiment "
            "component with skill_profile."
        )
    if physical.scene_config is None:
        raise ValueError(
            "Configured Task Program deployment must select or declare a "
            "physical scene."
        )

    deployment = _load_configured_task_program_deployment(
        task_program=task_program,
        skill_profile=physical.embodiment_skill_profile,
        evaluation=physical.evaluation,
        base_dir=deployment_path.parent,
    )
    _validate_scene_binding_targets(
        deployment.scene_binding,
        simulation=physical.scene_config,
    )

    from embodichain.lab.task_program.integrations.demo_skill import (
        _ConfiguredDemoSkillSource,
    )

    if (
        any(
            isinstance(source, _ConfiguredDemoSkillSource)
            for source in deployment.integration.adapter_factory.nominal_motion_sources
        )
        and physical.config.get("num_envs", 1) != 1
    ):
        raise ValueError("configured Demo Skills require num_envs == 1")
    registration = deployment.integration.registration
    registration.assert_unchanged()
    catalog = registration.catalog
    program = deployment.load_program()
    if program.program_id != deployment.program_id:
        raise ValueError(
            "Task integration expects program_id "
            f"{deployment.program_id!r}, got {program.program_id!r}."
        )
    if (
        isinstance(task_program, Mapping)
        and task_program.get("execution", "expert") == "policy"
    ):
        # Use the production decoder/validator, without mutating Gym's registry.
        from embodichain.lab.gym.utils.gym_utils import config_to_cfg

        config_to_cfg(dict(payload), source_path=deployment_path, _register_env=False)
    compiled = catalog.preflight(program, strict_contracts=strict_contracts)
    calls = tuple(
        compiled_call.call
        for segment in compiled.iter_segments()
        for compiled_call in segment.calls
    )
    semantic_calls = sorted({call.semantic_id for call in calls})

    environment_id = physical.config.get("id")
    if type(environment_id) is not str or not environment_id:
        raise ValueError("Runnable deployment must declare a non-empty string id.")
    summary = {
        "deployment": str(deployment_path),
        "environment_id": environment_id,
        "physics": physical.config.get("physics"),
        "program_id": program.program_id,
        "integration_id": deployment.integration_id,
        "scene_registry": deployment.selection.scene_registry,
        "robot_profile": deployment.selection.robot_profile,
        "runtime_preset": deployment.selection.runtime_preset,
        "execution": task_program.get("execution", "expert"),
        "action_owners": task_program.get("action_owners", {}),
        "segment_count": compiled.segment_count,
        "call_count": len(calls),
        "semantic_calls": semantic_calls,
    }
    if explain:
        from embodichain.lab.task_program.integrations.catalog import _canonical_value
        from embodichain.lab.task_program.integrations._program_identity import (
            task_program_fingerprint,
        )

        source_map = getattr(program, "__authoring_source_map__", None)
        summary["normalized_program"] = (
            None if source_map is None else source_map.declaration
        )
        summary["program_fingerprint"] = task_program_fingerprint(compiled)
        summary["contracts"] = catalog.semantic_planning_catalog(
            preset_id=deployment.selection.runtime_preset
        )
        summary["targets"] = _canonical_value(program.targets)
        summary["calls"] = [
            dict(
                index=c.call_index,
                segment=s.segment_id,
                source_path=list(c.source_path),
                authored_path=(
                    list(source_map.original_path(c.source_path))
                    if source_map
                    else list(c.source_path)
                ),
                call=c.call.to_metadata(),
                resource_ids={
                    **(
                        dict(
                            catalog.robot_profile.defaults[
                                catalog.call_catalog.discover(c.call).skill_id
                            ].resources
                        )
                        if catalog.call_catalog.discover(c.call).skill_id
                        in catalog.robot_profile.defaults
                        else {}
                    ),
                    **dict(c.call.resources),
                },
            )
            for s in compiled.iter_segments()
            for c in s.calls
        ]
        summary["nominal_sources"] = _canonical_value(
            deployment.integration.adapter_factory.nominal_motion_sources
        )
        summary["limitations"] = {
            "demo_skill_batch_size": 1,
            "configured_sources_require_sequential": True,
        }
        summary["resolved_configuration"] = _configuration_view(
            payload, physical, deployment, deployment_path
        )
    return summary


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Statically resolve and preflight configured Task Program deployments."
        )
    )
    parser.add_argument(
        "deployments",
        nargs="+",
        type=Path,
        help="Runnable task.<embodiment>.yaml deployment paths.",
    )
    parser.add_argument(
        "--strict-contracts",
        action="store_true",
        help="Require a structurally complete segment contract on every segment.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit one machine-readable result object.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Include resolved backend/parameters, authored owners, defaults and contracts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run static inspection for one or more deployments.

    Args:
        argv: Optional command-line arguments excluding the executable name.

    Returns:
        Zero when every deployment passes, otherwise one.
    """
    args = _parser().parse_args(argv)
    summaries: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for path in args.deployments:
        try:
            with redirect_stdout(sys.stderr):
                summaries.append(
                    inspect_deployment(
                        path,
                        explain=args.explain,
                        strict_contracts=args.strict_contracts,
                    )
                )
        except Exception as error:
            errors.append(
                {
                    "deployment": str(path.expanduser().resolve()),
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    if args.json:
        print(
            json.dumps(
                {"deployments": summaries, "errors": errors},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for summary in summaries:
            print(
                "[PASS] "
                f"{summary['environment_id']}: "
                f"{summary['segment_count']} segments, "
                f"{summary['call_count']} calls "
                f"({', '.join(summary['semantic_calls'])}); physics={summary['physics']}"
            )
            if args.explain:
                view = summary["resolved_configuration"]
                print(
                    "  Composed declarations and owners; --json also includes typed runtime defaults."
                )
                print(
                    f"  physics: {view['physics']['value']} <- {view['physics']['source']['file']}"
                )
                for name, owner in view["components"].items():
                    print(f"  {name}: {owner}")
                for resource in view["resources"]:
                    for endpoint in resource.get("endpoints", []):
                        print(
                            f"  resource.{resource['resource_id']}.{endpoint['endpoint_id']} "
                            f"-> {endpoint['control_part']} <- {view['resource_source']['file']}"
                        )
                for parameter in view["parameters"]:
                    source = parameter["source"]
                    field = ".".join([parameter["scope"], *parameter["path"]])
                    value = json.dumps(parameter["value"], ensure_ascii=False)
                    authored = ".".join(source["path"])
                    qualifier = " (derived)" if source["kind"] == "derived" else ""
                    print(
                        f"  {field} = {value} <- {source['file']}:{authored}{qualifier}"
                    )
                for parameter in view["monitor_parameters"]:
                    source = parameter["source"]
                    owner = (
                        f"{source['file']}:{'.'.join(source['path'])}"
                        if "file" in source
                        else f"SDK default ({source['type']})"
                    )
                    print(
                        f"  monitor.{parameter['monitor']}.{parameter['parameter']} "
                        f"= {parameter['value']} <- {owner}"
                    )
        for error in errors:
            print(
                "[FAIL] "
                f"{error['deployment']}: "
                f"{error['error_type']}: {error['message']}",
                file=sys.stderr,
            )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
